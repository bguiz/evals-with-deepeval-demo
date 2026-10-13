from deepeval import assert_test
from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Noul, Score
from deepeval.test_case import LLMTestCase, SingleTurnParams


def test_context_window_answer():
    correctness_metric = JevEval(
        name="Correctness",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        questions=[
            Noul(
                "No fact in actual_output contradicts a fact in expected_output.",
                weight=2,
            ),
            Score(
                "How much of the detail in expected_output is present in actual_output?",
                levels=["Almost none", "Some", "Most", "All of it"],
            ),
        ],
        threshold=0.5,
    )
    test_case = LLMTestCase(
        input="What is a context window?",
        # A hard-coded answer for now. Later tutorials use the agent's real answer.
        actual_output=(
            "A context window is the maximum number of tokens a model can "
            "process in one request, including the prompt and its reply."
        ),
        expected_output=(
            "A context window is the maximum number of tokens a model can "
            "process in one request. It includes the prompt, conversation "
            "history, retrieved information, tool results, and the model's "
            "reply. If the total exceeds the limit, content must be removed "
            "or the request may fail."
        ),
    )
    assert_test(test_case, [correctness_metric])
