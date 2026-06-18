# ABOUTME: LLM judge helpers for Pass@K evaluation quality logging.
# ABOUTME: Pure network/aggregation logic; no vLLM or trainer imports.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import time
from typing import Iterable, Optional, Sequence


JUDGE_RUBRIC = """You are evaluating a model response for general answer quality.

Score the response from 1 to 10 against the user's prompt. Consider:
- helpfulness and relevance to the prompt
- factual accuracy and internal consistency
- depth, completeness, and specificity
- creativity when the prompt calls for it
- conformance with explicit instructions and output constraints

Return only valid JSON in this exact shape:
{"Score": 8}
"""

_INT_RE = re.compile(r"-?\d+")


def build_judge_messages(prompt: str, response: str) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages for one judged prompt/response pair."""
    user_content = (
        "Prompt to evaluate against:\n"
        f"{prompt}\n\n"
        "Response to evaluate:\n"
        f"{response}\n\n"
        'Return JSON only, for example {"Score": 8}.'
    )
    return [
        {"role": "system", "content": JUDGE_RUBRIC},
        {"role": "user", "content": user_content},
    ]


def _coerce_score(value) -> Optional[int]:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, min(10, score))


def parse_score(text: str | None) -> Optional[int]:
    """Parse a 1-10 judge score from JSON, with a small integer fallback."""
    if text is None:
        return None

    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        parsed = None

    if isinstance(parsed, dict):
        score = _coerce_score(parsed.get("Score"))
        if score is not None:
            return score
    elif parsed is not None:
        score = _coerce_score(parsed)
        if score is not None:
            return score

    match = _INT_RE.search(str(text))
    if not match:
        return None
    return _coerce_score(match.group(0))


def aggregate_quality(
    quality: Sequence[Optional[int]],
    correctness: Sequence[Optional[bool]],
    *,
    conditioned: bool = True,
) -> dict[str, float | int | None]:
    """Aggregate judge scores, excluding failed judge calls from quality means."""
    valid_pairs = [
        (score, correct)
        for score, correct in zip(quality, correctness)
        if score is not None
    ]
    valid_scores = [score for score, _ in valid_pairs]

    def _mean(values: Sequence[int]) -> float | None:
        return (sum(values) / len(values)) if values else None

    metrics: dict[str, float | int | None] = {
        "quality": _mean(valid_scores),
        "quality_n_judged": len(valid_scores),
        "quality_n_judge_failures": len(quality) - len(valid_scores),
    }
    if not conditioned:
        return metrics

    correct_scores = [score for score, correct in valid_pairs if correct is True]
    incorrect_scores = [score for score, correct in valid_pairs if correct is False]
    metrics.update({
        "quality_when_correct": _mean(correct_scores),
        "quality_when_incorrect": _mean(incorrect_scores),
        "quality_n_correct": len(correct_scores),
        "quality_n_incorrect": len(incorrect_scores),
    })
    return metrics


class DeepSeekJudge:
    """Concurrent DeepSeek V4 Flash judge client."""

    def __init__(
        self,
        *,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        concurrency: int = 16,
        timeout: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"{api_key_env} is required when the Pass@K judge is enabled"
            )

        from openai import OpenAI

        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def score_pairs(self, pairs: Iterable[tuple[str, str]]) -> list[Optional[int]]:
        """Score prompt/response pairs, preserving input order."""
        pairs = list(pairs)
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(self._score_one, pairs))

    def _score_one(self, pair: tuple[str, str]) -> Optional[int]:
        prompt, response = pair
        for attempt in range(self.max_retries + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=build_judge_messages(prompt, response),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = completion.choices[0].message.content
                return parse_score(content)
            except Exception as exc:
                if attempt >= self.max_retries or not _is_retryable(exc):
                    return None
                time.sleep(min(2 ** attempt, 8))
        return None


def _is_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    return exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }
