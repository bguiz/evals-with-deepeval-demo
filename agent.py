"""A minimal ReAct agent: reads queries from stdin, writes progress and answers to stdout.

No agent framework, just the OpenAI SDK (Responses API). The agent loop, LLM
calls and tool calls are traced as deepeval spans with @observe, so evals can
inspect what happens inside a run, not just the final answer. Each LLM call is
constrained to a JSON schema describing one ReAct step: a thought, then either
an action (keyword search of the knowledge base) or a final answer. The loop runs the
action, feeds the observation back, and repeats until the model answers.

Configuration (from .env):
    INFERENCE_BASE_URL    API root, e.g. https://.../v1 (the SDK appends /responses)
    INFERENCE_API_KEY     API key for that endpoint
    INFERENCE_MODEL_NAME  model id

Usage:
    uv run agent.py            # one query per line; /exit or Ctrl-D to quit
    uv run agent.py -v         # also print the ReAct thoughts to stderr
    echo "What is RAG?" | uv run agent.py
"""

import json
import os
import re
import sys
from pathlib import Path

# Traces are only for evals; don't print Confident AI trace-posting notices on
# normal runs. Set before importing deepeval, which reads it at startup.
os.environ.setdefault("CONFIDENT_TRACE_VERBOSE", "0")

from deepeval.tracing import observe, update_current_span  # noqa: E402
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

KNOWLEDGE_PATH = Path(__file__).parent / "agent-knowledge.md"
MODEL = os.environ["INFERENCE_MODEL_NAME"]
MAX_USER_TURNS = 5  # context window: keep only the last 5 user messages
MAX_STEPS = 5  # ReAct steps per query, so a confused model can't loop forever
MAX_OUTPUT_TOKENS = 1024  # per LLM call; reasoning models count reasoning tokens here too
MAX_SEARCH_RESULTS = 8  # paragraphs per search, so a broad keyword can't return the whole file
VERBOSE = "-v" in sys.argv[1:]

SYSTEM_PROMPT = """You are a helpful assistant that answers beginner questions about AI engineering.

You work in a ReAct loop. Each reply is one step, as JSON:
- "thought": brief reasoning about what to do next.
- "action": either "search_knowledge" or "final_answer".
- "keywords": for "search_knowledge", 1-5 single-word keywords to search for, otherwise [].
- "answer": the answer for the user when action is "final_answer", otherwise "".

The "search_knowledge" action does a keyword search of the knowledge base and
returns the best-matching paragraphs (up to a limit) with the headings they
sit under, as an observation. Paragraphs matching more keywords rank higher. A paragraph matches if it contains any keyword; a
keyword also matches words that start with it, so "token" matches "tokens".
Prefer specific terms and synonyms (e.g. "RAG", "retrieval") over generic
words like "model". If the results are not enough, search again with
different keywords.

Search before answering any question about AI engineering, unless the relevant
passages are already present in this conversation. Base your answers on the
knowledge base. If it does not cover the question, say so rather than guessing.

If any query is not found in the results of "search_knowledge", answer exactly "Sorry I cannot help with that."

Keep answers concise: 2-5 sentences, in plain text."""

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["search_knowledge", "final_answer"]},
        # Enforces the tool's contract during decoding: at most 5 keywords, each
        # a single word (no whitespace). Empty for final_answer, so no minItems.
        "keywords": {
            "type": "array",
            "items": {"type": "string", "pattern": "^\\S+$"},
            "maxItems": 5,
        },
        "answer": {"type": "string"},
    },
    "required": ["thought", "action", "keywords", "answer"],
    "additionalProperties": False,
}


def load_paragraphs(path: Path) -> list[tuple[list[str], str]]:
    """Split a Markdown file into (headings it sits under, paragraph text) pairs."""
    paragraphs = []
    headings: list[tuple[int, str]] = []  # (level, heading line), outermost first
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        block = block.strip()
        heading = re.match(r"(#+)\s", block)
        if heading:
            level = len(heading.group(1))
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, block))
        elif block:
            paragraphs.append(([line for _, line in headings], block))
    return paragraphs


@observe(type="tool")
def search_knowledge(keywords: list[str]) -> str:
    """The agent's only tool: keyword search over the knowledge base paragraphs."""
    # The span records the keywords exactly as the model sent them, as JSON.
    update_current_span(input=to_json({"keywords": keywords}))
    # Phrases are split into words, so "RAG chunking" searches for "RAG" and "chunking".
    keywords = list(dict.fromkeys(word for k in keywords for word in k.split()))
    if not keywords:
        return "No keywords given. Provide 1-5 keywords to search for."

    # Match at the start of a word, case-insensitively, so "token" also matches "tokens".
    patterns = [re.compile(r"\b" + re.escape(k), re.IGNORECASE) for k in keywords]
    matches = []
    for position, (headings, text) in enumerate(load_paragraphs(KNOWLEDGE_PATH)):
        hits = sum(1 for pattern in patterns if pattern.search(text))
        if hits:
            matches.append((hits, position, headings, text))
    if not matches:
        return f"No paragraphs matched: {', '.join(keywords)}"

    # Keep the paragraphs matching the most keywords, then show them in file order.
    top = sorted(matches, key=lambda m: (-m[0], m[1]))[:MAX_SEARCH_RESULTS]
    results = [
        "\n".join(headings + [text]) for _, _, headings, text in sorted(top, key=lambda m: m[1])
    ]
    summary = f"{len(matches)} paragraphs matched; showing {len(top)}."
    return "\n\n---\n\n".join([summary] + results)


def make_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["INFERENCE_BASE_URL"],
        api_key=os.environ["INFERENCE_API_KEY"],
    )


@observe(type="llm", model=MODEL)
def call_llm(client: OpenAI, messages: list[dict]) -> dict:
    update_current_span(input=messages)  # rather than the default, which includes the client
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=messages,
        text={
            "format": {
                "type": "json_schema",
                "name": "react_step",
                "schema": STEP_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
    )
    if response.status == "incomplete":
        raise RuntimeError(f"LLM response incomplete: {response.incomplete_details}")
    return json.loads(response.output_text)


def build_context(history: list[list[dict]]) -> list[dict]:
    """The last MAX_USER_TURNS turns, flattened into messages.

    Each turn starts with a user message, followed by that query's AI steps and
    tool observations, so dropping a turn drops all of them together.
    """
    return [message for turn in history[-MAX_USER_TURNS:] for message in turn]


def log(text: str) -> None:
    if VERBOSE:
        print(text, file=sys.stderr, flush=True)


def show(text: str) -> None:
    """Progress output on stdout, alongside the answers."""
    print(text, flush=True)


def truncate_values(value, limit: int = 50):
    """Shorten every string longer than `limit` to `limit` characters plus "…"."""
    if isinstance(value, str):
        return value[:limit] + "…" if len(value) > limit else value
    if isinstance(value, list):
        return [truncate_values(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: truncate_values(item, limit) for key, item in value.items()}
    return value


def to_json(value) -> str:
    # Like JS JSON.stringify: compact, and non-ASCII characters left as-is.
    return json.dumps(value, ensure_ascii=False)


@observe(type="agent")
def answer_query(client: OpenAI, history: list[list[dict]], query: str) -> str:
    update_current_span(input=query)
    turn = [{"role": "user", "content": query}]
    history.append(turn)
    del history[:-MAX_USER_TURNS]  # older turns can never be sent again

    try:
        for _ in range(MAX_STEPS):
            show("* Flibbertigibbeting")
            step = call_llm(client, build_context(history))
            turn.append({"role": "assistant", "content": json.dumps(step)})
            log(f"[thought] {step['thought']}\n[action] {step['action']}")

            if step["action"] == "final_answer":
                return step["answer"]

            show(f"> > tool: search_knowledge {to_json({'keywords': step['keywords']})}")
            observation = search_knowledge(step["keywords"])
            # Display only: the model still gets the full observation.
            show(f"< < tool: search_knowledge {to_json(truncate_values(observation))}")
            log(f"[keywords] {step['keywords']}\n[observation] {observation.splitlines()[0]}")
            turn.append({"role": "user", "content": f"Observation from search_knowledge:\n{observation}"})
    except Exception:
        history.remove(turn)  # don't keep a half-finished turn in the context
        raise

    return f"Sorry, I couldn't reach an answer within {MAX_STEPS} steps."


def main() -> None:
    client = make_client()
    history: list[list[dict]] = []
    interactive = sys.stdin.isatty()

    while True:
        if interactive:
            print("> ", end="", file=sys.stderr, flush=True)
        line = sys.stdin.readline()
        if not line:  # EOF
            break
        query = line.strip()
        if not query:
            continue
        if query == "/exit":
            break
        try:
            answer = answer_query(client, history, query)
        except (OpenAIError, RuntimeError, json.JSONDecodeError) as e:
            print(f"error: {e}", file=sys.stderr, flush=True)
            continue
        show(f"< {answer}")


if __name__ == "__main__":
    main()
