"""Evals for agent.py, one layer at a time.

Run with `uv run deepeval test run test_agent.py`: the structured layer needs
the per-test trace that DeepEval's pytest plugin sets up.

Layers, cheapest first:
1. structured: rule-based validation of the agent's tool calls
2. fuzzy:      key-phrase matching of the answer (exact, RapidFuzz, BLEU/ROUGE)
3. judge:      Jev checks the answer against a reference answer, one API call per test
"""

from collections import Counter
from pathlib import Path

import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import JevEval, ToolCorrectnessMetric
from deepeval.metrics.jev_eval import Noul, Score
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.tracing import observe, update_current_span

import agent
from deepeval_fuzzy_util import FuzzyMatchMetric
from deepeval_jev_util import JevQuestionMetric, SharedJevDecision
from deepeval_tool_util import ExpectedVerdictMetric, attach_tool_metrics, record_tools_called

# Toggle each layer of evals.
RUN_STRUCTURED_EVALS = True  # validation of the agent's tool calls, via @observe spans
RUN_FUZZY_EVALS = True  # key-phrase matching of the answer: exact, RapidFuzz, BLEU/ROUGE
RUN_JEV_EVALS = True  # the judge: Jev compares the answer with a reference answer

GOLDENS_DIR = Path(__file__).parent / "goldens"


def load_goldens(file_name: str) -> list[Golden]:
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(GOLDENS_DIR / file_name))
    return dataset.goldens


# Each layer has expected passes, run by the live agent, and expected failures,
# run by a mocked agent. Only the golden sets of enabled layers are loaded; a
# disabled layer's tests get an empty parameter list and are skipped.
structured_pass_goldens = (
    load_goldens("agent_structured_pass_goldens.json") if RUN_STRUCTURED_EVALS else []
)
structured_fail_goldens = (
    load_goldens("agent_structured_fail_goldens.json") if RUN_STRUCTURED_EVALS else []
)
# expected_output holds key phrases, one requirement per line, instead of a reference answer.
fuzzy_pass_goldens = load_goldens("agent_fuzzy_pass_goldens.json") if RUN_FUZZY_EVALS else []
fuzzy_fail_goldens = load_goldens("agent_fuzzy_fail_goldens.json") if RUN_FUZZY_EVALS else []
# expected_output holds a reference answer.
jev_pass_goldens = load_goldens("agent_judge_pass_goldens.json") if RUN_JEV_EVALS else []
jev_fail_goldens = load_goldens("agent_judge_fail_goldens.json") if RUN_JEV_EVALS else []

# Failure modes each layer is expected to catch.
STRUCTURED_FAILURE_MODES = ["fail_invalid_keywords", "fail_irrelevant_results", "fail_no_tool_call"]
# Fuzzy matching checks which concepts an answer covers, not whether what it
# says about them is true, so contradictions are left to the judge.
FUZZY_FAILURE_MODES = ["fail_incomplete"]
JEV_FAILURE_MODES = ["fail_contradiction", "fail_incomplete"]

golden_set_pairs = {
    name: sets
    for name, sets, enabled in [
        (
            "structured",
            (structured_pass_goldens, structured_fail_goldens, STRUCTURED_FAILURE_MODES),
            RUN_STRUCTURED_EVALS,
        ),
        ("fuzzy", (fuzzy_pass_goldens, fuzzy_fail_goldens, FUZZY_FAILURE_MODES), RUN_FUZZY_EVALS),
        ("judge", (jev_pass_goldens, jev_fail_goldens, JEV_FAILURE_MODES), RUN_JEV_EVALS),
    ]
    if enabled
}

# Whether each metric should pass for each scenario; None means not checked.
STRUCTURED_EXPECTED_OUTCOMES = {
    "pass": {"Tool call validation": True, "Tool correctness": True},
    "fail_invalid_keywords": {"Tool call validation": False, "Tool correctness": True},
    "fail_irrelevant_results": {"Tool call validation": False, "Tool correctness": True},
    "fail_no_tool_call": {"Tool call validation": None, "Tool correctness": False},
}

# (metric name, Jev question). All questions go to Jev in a single call, and
# each one is reported as its own metric.
QUESTIONS = [
    (
        "No contradiction",
        Noul("No fact in actual_output contradicts a fact in expected_output."),
    ),
    (
        "Completeness",
        Score(
            "How much of the detail in expected_output is present in actual_output?",
            levels=["Almost none", "Some", "Most", "All of it"],
        ),
    ),
]

# Whether each judge metric should pass for each scenario. A metric missing
# from a scenario is reported with its score, but doesn't decide pass/fail.
EXPECTED_OUTCOMES = {
    "pass": {"No contradiction": True, "Completeness": True},
    "fail_contradiction": {"No contradiction": False},
    "fail_incomplete": {"No contradiction": True, "Completeness": False},
}

skip_structured = pytest.mark.skipif(not RUN_STRUCTURED_EVALS, reason="RUN_STRUCTURED_EVALS is False")
skip_fuzzy = pytest.mark.skipif(not RUN_FUZZY_EVALS, reason="RUN_FUZZY_EVALS is False")
skip_jev = pytest.mark.skipif(not RUN_JEV_EVALS, reason="RUN_JEV_EVALS is False")


def golden_id(golden: Golden) -> str:
    meta = golden.additional_metadata
    return f"q{meta['question_id']}-{meta['scenario']}"


_answers: dict[str, str] = {}


def run_agent(query: str, fresh: bool = False) -> str:
    """agent.py's answer to a single query, with no earlier conversation.

    Answers are cached, so each query is answered once per run even when
    several layers use it. `fresh=True` always runs the agent, which the
    structured layer needs: a cached answer records no spans in the current
    test's trace.
    """
    if fresh or query not in _answers:
        _answers[query] = agent.answer_query(agent.make_client(), history=[], query=query)
    return _answers[query]


def mocked_agent(golden: Golden):
    """An agent that simulates the golden's failure mode by giving its incorrect answer."""

    def answer(query: str) -> str:
        assert query == golden.input
        return golden.actual_output

    return answer


@observe(type="tool", name="search_knowledge")
def replay_tool_call(keywords: list[str], output: str) -> str:
    update_current_span(input=agent.to_json({"keywords": keywords}), output=output)
    return output


@observe(type="agent", name="mocked_agent")
def replay_tool_calls(golden: Golden) -> str:
    """A mocked agent that simulates the golden's failure mode by making its
    recorded tool calls, traced as tool spans just like agent.py's."""
    update_current_span(input=golden.input)
    for call in golden.tools_called or []:
        replay_tool_call(call.input_parameters["keywords"], call.output)
    return "(mocked answer)"


def assert_tool_calls(golden: Golden) -> None:
    # Each tool span in the test's trace gets a validation metric, and the
    # trace gets a tool correctness metric comparing tools_called with
    # the golden's expected_tools.
    expected = STRUCTURED_EXPECTED_OUTCOMES[golden.additional_metadata["scenario"]]
    attach_tool_metrics(
        max_results=agent.MAX_SEARCH_RESULTS, expected_pass=expected["Tool call validation"]
    )
    record_tools_called()
    tool_correctness = ExpectedVerdictMetric(
        ToolCorrectnessMetric(threshold=1.0),
        expected_pass=expected["Tool correctness"],
        name="Tool correctness",
    )
    assert_test(golden=golden, metrics=[tool_correctness])


def make_test_case(golden: Golden, actual_output: str) -> LLMTestCase:
    return LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        context=golden.context,
    )


def assert_fuzzy_match(golden: Golden, actual_output: str) -> None:
    expected_pass = golden.additional_metadata["expected_verdict"] == "pass"
    assert_test(make_test_case(golden, actual_output), [FuzzyMatchMetric(expected_pass=expected_pass)])


def assert_jev_correctness(golden: Golden, actual_output: str) -> None:
    decision = SharedJevDecision(
        JevEval(
            name="Correctness",
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
            questions=[question for _, question in QUESTIONS],
        )
    )
    expected = EXPECTED_OUTCOMES[golden.additional_metadata["scenario"]]
    metrics = [
        JevQuestionMetric(decision, i, name, expected_pass=expected.get(name))
        for i, (name, _) in enumerate(QUESTIONS)
    ]
    assert_test(make_test_case(golden, actual_output), metrics)


@pytest.mark.parametrize(
    "pass_goldens, fail_goldens, failure_modes",
    list(golden_set_pairs.values()),
    ids=list(golden_set_pairs),
)
def test_golden_sets_aligned(
    pass_goldens: list[Golden], fail_goldens: list[Golden], failure_modes: list[str]
):
    """Every pass query appears exactly once per failure mode in the fail set."""
    pass_inputs = Counter(g.input for g in pass_goldens)
    assert all(n == 1 for n in pass_inputs.values()), "duplicate queries in the pass set"
    fail_pairs = Counter((g.input, g.additional_metadata["scenario"]) for g in fail_goldens)
    assert fail_pairs == Counter({(i, s): 1 for i in pass_inputs for s in failure_modes})


# Structured validation of tool calls


@skip_structured
@pytest.mark.parametrize("golden", structured_pass_goldens, ids=golden_id)
def test_agent_tool_calls(golden: Golden):
    # Runs inside the test's DeepEval trace, so the agent's spans are recorded there.
    run_agent(golden.input, fresh=True)
    assert_tool_calls(golden)


@skip_structured
@pytest.mark.parametrize("golden", structured_fail_goldens, ids=golden_id)
def test_mocked_agent_tool_calls(golden: Golden):
    replay_tool_calls(golden)
    assert_tool_calls(golden)


# Fuzzy matching of the answer. Defined after the structured tests, which run
# first and fill the answer cache, so the agent answers each query only once.


@skip_fuzzy
@pytest.mark.parametrize("golden", fuzzy_pass_goldens, ids=golden_id)
def test_agent_fuzzy(golden: Golden):
    assert_fuzzy_match(golden, run_agent(golden.input))


@skip_fuzzy
@pytest.mark.parametrize("golden", fuzzy_fail_goldens, ids=golden_id)
def test_mocked_agent_failures_fuzzy(golden: Golden):
    assert_fuzzy_match(golden, mocked_agent(golden)(golden.input))


# The judge. Defined last: it's the slowest and costliest layer, and by now
# the answer cache is full.


@skip_jev
@pytest.mark.parametrize("golden", jev_pass_goldens, ids=golden_id)
def test_agent_jev(golden: Golden):
    assert_jev_correctness(golden, run_agent(golden.input))


@skip_jev
@pytest.mark.parametrize("golden", jev_fail_goldens, ids=golden_id)
def test_mocked_agent_failures_jev(golden: Golden):
    assert_jev_correctness(golden, mocked_agent(golden)(golden.input))
