# ABOUTME: Held-out math evaluation split drawn from MATH-500 and the GSM8K test set.
# ABOUTME: Neither source seeds OpenMathInstruct-2, so its problems never appear in training.

from datasets import Dataset, load_dataset

from tuning.data.config import COMPMATH_STRING, SYSTEM_MESSAGE_OPENMATH

MATH500_PATH = "HuggingFaceH4/MATH-500"
GSM8K_PATH = "openai/gsm8k"
GSM8K_ANSWER_MARKER = "####"


def boxed_gsm8k_solution(solution: str) -> str:
    """Rewrite GSM8K's '#### 42' tail into the $\\boxed{42}$ form the system prompt asks for."""
    reasoning, marker, answer = solution.rpartition(GSM8K_ANSWER_MARKER)
    if not marker:
        return solution
    return f"{reasoning.strip()}\n$\\boxed{{{answer.strip()}}}$"


def to_openmath_row(problem: str, solution: str) -> dict:
    """Render one problem/solution pair the way the openmath SFT builder renders training rows."""
    prompt = COMPMATH_STRING.format(problem=problem)
    return {
        "prompt": prompt,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": solution},
        ],
    }


def build_heldout_math_eval() -> Dataset:
    """Build the evaluation split: MATH-500 competition problems plus GSM8K grade-school problems."""
    rows = [
        to_openmath_row(row["problem"], row["solution"])
        for row in load_dataset(MATH500_PATH, split="test")
    ]
    rows += [
        to_openmath_row(row["question"], boxed_gsm8k_solution(row["answer"]))
        for row in load_dataset(GSM8K_PATH, "main", split="test")
    ]
    return Dataset.from_list(rows)
