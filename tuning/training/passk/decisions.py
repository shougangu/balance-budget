# ABOUTME: Pure-logic engine that decides when to save sweetspot checkpoints.
# ABOUTME: Owns threshold, early-tuple, fixed target, max-gap decisions — no W&B / model deps.

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class CheckpointDecision:
    label: str
    advances_state: bool
    metadata_type: Optional[str] = None
    metadata_value: Any = None
    eval_only: bool = False


class CheckpointDecisionEngine:
    def __init__(
        self,
        target_thresholds: List[float],
        early_tuples: Optional[List[Tuple[int, float]]],
        max_checkpoint_gap: Optional[int],
        target_data_points: Optional[List[int]] = None,
        target_total_minutes: Optional[List[float]] = None,
        eval_only_minutes: Optional[List[float]] = None,
    ):
        self.target_thresholds = sorted(target_thresholds, reverse=True)
        self.early_tuples = list(early_tuples) if early_tuples else None
        self.max_checkpoint_gap = max_checkpoint_gap
        self.target_data_points = (
            sorted(target_data_points) if target_data_points else None
        )
        self.target_total_minutes = (
            sorted(target_total_minutes) if target_total_minutes else None
        )
        self.eval_only_minutes = set(eval_only_minutes or [])
        self.pending_total_minute_targets: List[float] = []

    def consume_crossed_total_minute_targets(
        self, total_minutes: float,
    ) -> List[float]:
        if not self.target_total_minutes:
            return []
        crossed = [t for t in self.target_total_minutes if t <= total_minutes]
        if crossed:
            self.target_total_minutes = self.target_total_minutes[len(crossed):]
            self.pending_total_minute_targets.extend(crossed)
        return crossed

    def decide(
        self,
        primary_metric: float,
        history: List[float],
        data_points_seen: int,
        last_checkpoint_data_points: int,
        total_minutes: Optional[float] = None,
    ) -> List[CheckpointDecision]:
        decisions: List[CheckpointDecision] = []

        if self.target_thresholds:
            reached_index = None
            reached_threshold = None
            for i, threshold in enumerate(self.target_thresholds):
                if primary_metric >= threshold:
                    reached_index = i
                    reached_threshold = threshold
                    break
            if reached_threshold is not None:
                decisions.append(CheckpointDecision(
                    label=str(reached_threshold), advances_state=True
                ))
                self.target_thresholds = self.target_thresholds[:reached_index]

        if self.early_tuples is not None:
            triggered_idx = []
            for idx, (patience, min_increase) in enumerate(self.early_tuples):
                if len(history) > patience:
                    early_stopping = True
                    for old, new in zip(history[-patience-1:], history[-patience:]):
                        if new - old >= min_increase:
                            early_stopping = False
                            break
                    if early_stopping:
                        decisions.append(CheckpointDecision(
                            label=f"{patience}@{min_increase}",
                            advances_state=True,
                        ))
                        triggered_idx.append(idx)
            for idx in reversed(triggered_idx):
                self.early_tuples.pop(idx)

        if self.target_data_points:
            crossed = [t for t in self.target_data_points
                       if t <= data_points_seen]
            if crossed:
                decisions.append(CheckpointDecision(
                    label=f"data-{crossed[-1]}",
                    advances_state=True,
                ))
                self.target_data_points = self.target_data_points[len(crossed):]

        if total_minutes is not None:
            self.consume_crossed_total_minute_targets(total_minutes)
        if self.pending_total_minute_targets:
            pending = self.pending_total_minute_targets
            self.pending_total_minute_targets = []
            # Emit a separate decision per kind so an eval-only crossing never
            # collapses/suppresses a real (GRPO-bound) checkpoint when both
            # cross in the same eval.
            for eval_only in (False, True):
                group = [t for t in pending
                         if (t in self.eval_only_minutes) == eval_only]
                if group:
                    target_minutes = max(group)
                    decisions.append(CheckpointDecision(
                        label=f"{target_minutes:g}m",
                        advances_state=True,
                        metadata_type="total_minutes",
                        metadata_value=target_minutes,
                        eval_only=eval_only,
                    ))

        if (self.max_checkpoint_gap is not None
                and data_points_seen > 0
                and not decisions):
            gap = data_points_seen - last_checkpoint_data_points
            if gap >= self.max_checkpoint_gap:
                decisions.append(CheckpointDecision(
                    label=f"gap-{data_points_seen}-{primary_metric}",
                    advances_state=True,
                ))

        return decisions
