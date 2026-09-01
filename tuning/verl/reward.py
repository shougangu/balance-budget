# ABOUTME: verl custom reward for math RLVR: 1.0 when the union grader accepts the answer.
# ABOUTME: Same verifier as offline eval (lm-eval numeric OR math-verify), so reward == metric.

from tuning.evaluation.math_scoring import is_correct


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    # verl scores from worker threads, where math-verify's signal-based
    # timeout cannot be armed.
    return 1.0 if is_correct(solution_str, ground_truth, timeout_seconds=None) else 0.0
