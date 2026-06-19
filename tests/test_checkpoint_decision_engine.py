# ABOUTME: Unit tests for CheckpointDecisionEngine — pure threshold/early-tuple/gap logic.
# ABOUTME: No vllm or wandb mocks needed (engine has zero such deps).

from tuning.training.passk.decisions import CheckpointDecision, CheckpointDecisionEngine


def _engine(thresholds=None, early_tuples=None, max_gap=None,
            target_data_points=None, target_total_minutes=None,
            eval_only_minutes=None):
    return CheckpointDecisionEngine(
        target_thresholds=thresholds or [],
        early_tuples=early_tuples or None,
        max_checkpoint_gap=max_gap,
        target_data_points=target_data_points,
        target_total_minutes=target_total_minutes,
        eval_only_minutes=eval_only_minutes,
    )


class TestThresholdSweep:
    def test_sorted_descending_on_init(self):
        eng = _engine(thresholds=[0.3, 0.7, 0.5])
        assert eng.target_thresholds == [0.7, 0.5, 0.3]

    def test_no_decision_when_below_all(self):
        eng = _engine(thresholds=[0.7, 0.5, 0.3])
        decisions = eng.decide(primary_metric=0.2, history=[0.2],
                               data_points_seen=100, last_checkpoint_data_points=0)
        assert decisions == []

    def test_picks_hardest_reached_and_trims(self):
        eng = _engine(thresholds=[0.7, 0.5, 0.3])
        decisions = eng.decide(primary_metric=0.55, history=[0.55],
                               data_points_seen=100, last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(label="0.5", advances_state=True)]
        assert eng.target_thresholds == [0.7]

    def test_subsequent_call_after_threshold_consumed(self):
        eng = _engine(thresholds=[0.7, 0.5, 0.3])
        eng.decide(primary_metric=0.55, history=[0.55],
                   data_points_seen=100, last_checkpoint_data_points=0)
        decisions = eng.decide(primary_metric=0.55, history=[0.55, 0.55],
                               data_points_seen=200, last_checkpoint_data_points=100)
        assert decisions == []


class TestEarlyTuples:
    def test_no_trigger_when_history_too_short(self):
        eng = _engine(early_tuples=[(2, 0.05)])
        decisions = eng.decide(primary_metric=0.5, history=[0.5, 0.5],
                               data_points_seen=100, last_checkpoint_data_points=0)
        assert decisions == []

    def test_triggers_when_no_increase_over_window(self):
        eng = _engine(early_tuples=[(2, 0.05)])
        decisions = eng.decide(primary_metric=0.5,
                               history=[0.5, 0.5, 0.5],
                               data_points_seen=100, last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(label="2@0.05", advances_state=True)]
        assert eng.early_tuples == []

    def test_does_not_trigger_when_increase_seen(self):
        eng = _engine(early_tuples=[(2, 0.05)])
        decisions = eng.decide(primary_metric=0.6,
                               history=[0.4, 0.5, 0.6],
                               data_points_seen=100, last_checkpoint_data_points=0)
        assert decisions == []
        assert eng.early_tuples == [(2, 0.05)]


class TestGapCheckpoint:
    def test_no_gap_when_disabled(self):
        eng = _engine(max_gap=None)
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=10000, last_checkpoint_data_points=0)
        assert decisions == []

    def test_gap_triggers_when_distance_exceeds_max(self):
        eng = _engine(max_gap=5000)
        decisions = eng.decide(primary_metric=0.42, history=[0.42],
                               data_points_seen=6000, last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(label="gap-6000-0.42", advances_state=True)]

    def test_gap_skipped_when_threshold_already_fired(self):
        eng = _engine(thresholds=[0.5], max_gap=1000)
        decisions = eng.decide(primary_metric=0.6,
                               history=[0.6],
                               data_points_seen=2000, last_checkpoint_data_points=0)
        assert len(decisions) == 1
        assert decisions[0].label == "0.5"


class TestFixedDataTargets:
    def test_no_trigger_below_first_target(self):
        eng = _engine(target_data_points=[4000, 8000])
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=2000, last_checkpoint_data_points=0)
        assert decisions == []
        assert eng.target_data_points == [4000, 8000]

    def test_fires_at_first_crossing(self):
        eng = _engine(target_data_points=[4000, 8000])
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=4500, last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(label="data-4000",
                                                advances_state=True)]
        assert eng.target_data_points == [8000]

    def test_target_consumed_after_firing(self):
        eng = _engine(target_data_points=[4000, 8000])
        eng.decide(primary_metric=0.0, history=[0.0],
                   data_points_seen=4500, last_checkpoint_data_points=0)
        decisions = eng.decide(primary_metric=0.0, history=[0.0, 0.0],
                               data_points_seen=5000,
                               last_checkpoint_data_points=4500)
        assert decisions == []

    def test_multiple_crossings_one_eval_picks_highest(self):
        eng = _engine(target_data_points=[4000, 8000])
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=8500, last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(label="data-8000",
                                                advances_state=True)]
        assert eng.target_data_points == []

    def test_unsorted_input_handled(self):
        eng = _engine(target_data_points=[12000, 4000, 8000])
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=4500, last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(label="data-4000",
                                                advances_state=True)]
        assert eng.target_data_points == [8000, 12000]

    def test_fires_alongside_threshold(self):
        eng = _engine(thresholds=[0.5], target_data_points=[4000])
        decisions = eng.decide(primary_metric=0.6, history=[0.6],
                               data_points_seen=4500, last_checkpoint_data_points=0)
        labels = sorted(d.label for d in decisions)
        assert labels == ["0.5", "data-4000"]
        assert all(d.advances_state for d in decisions)
        assert eng.target_thresholds == []
        assert eng.target_data_points == []

    def test_suppresses_gap_when_fired(self):
        eng = _engine(target_data_points=[4000], max_gap=1000)
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=4500, last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(label="data-4000",
                                                advances_state=True)]

    def test_none_means_disabled(self):
        eng = _engine(target_data_points=None)
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=999999,
                               last_checkpoint_data_points=0)
        assert decisions == []


class TestFixedTotalMinuteTargets:
    def test_no_trigger_below_first_target(self):
        eng = _engine(target_total_minutes=[30.0, 60.0])
        crossed = eng.consume_crossed_total_minute_targets(29.9)
        assert crossed == []
        assert eng.target_total_minutes == [30.0, 60.0]
        assert eng.pending_total_minute_targets == []

    def test_step_end_consumes_crossed_target_and_defers_decision(self):
        eng = _engine(target_total_minutes=[30.0, 60.0])
        crossed = eng.consume_crossed_total_minute_targets(30.1)
        assert crossed == [30.0]
        assert eng.target_total_minutes == [60.0]
        assert eng.pending_total_minute_targets == [30.0]

    def test_pending_target_fires_on_decide(self):
        eng = _engine(target_total_minutes=[30.0])
        eng.consume_crossed_total_minute_targets(30.1)
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=4000,
                               last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(
            label="30m",
            advances_state=True,
            metadata_type="total_minutes",
            metadata_value=30.0,
        )]
        assert eng.pending_total_minute_targets == []

    def test_decide_can_consume_total_minutes_directly(self):
        eng = _engine(target_total_minutes=[30.0])
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=4000,
                               last_checkpoint_data_points=0,
                               total_minutes=30.1)
        assert decisions == [CheckpointDecision(
            label="30m",
            advances_state=True,
            metadata_type="total_minutes",
            metadata_value=30.0,
        )]
        assert eng.target_total_minutes == []

    def test_multiple_crossings_one_eval_picks_highest(self):
        eng = _engine(target_total_minutes=[30.0, 60.0])
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=4000,
                               last_checkpoint_data_points=0,
                               total_minutes=61.0)
        assert decisions == [CheckpointDecision(
            label="60m",
            advances_state=True,
            metadata_type="total_minutes",
            metadata_value=60.0,
        )]


class TestEvalOnlyMinutes:
    def test_regular_target_is_not_eval_only(self):
        eng = _engine(target_total_minutes=[30.0])
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=4000,
                               last_checkpoint_data_points=0,
                               total_minutes=30.1)
        assert decisions == [CheckpointDecision(
            label="30m",
            advances_state=True,
            metadata_type="total_minutes",
            metadata_value=30.0,
            eval_only=False,
        )]

    def test_eval_only_target_flagged(self):
        eng = _engine(target_total_minutes=[30.0], eval_only_minutes=[30.0])
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=4000,
                               last_checkpoint_data_points=0,
                               total_minutes=30.1)
        assert decisions == [CheckpointDecision(
            label="30m",
            advances_state=True,
            metadata_type="total_minutes",
            metadata_value=30.0,
            eval_only=True,
        )]

    def test_mixed_crossing_emits_regular_and_eval_only(self):
        eng = _engine(target_total_minutes=[30.0, 60.0],
                      eval_only_minutes=[30.0])
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=4000,
                               last_checkpoint_data_points=0,
                               total_minutes=61.0)
        assert decisions == [
            CheckpointDecision(
                label="60m",
                advances_state=True,
                metadata_type="total_minutes",
                metadata_value=60.0,
                eval_only=False,
            ),
            CheckpointDecision(
                label="30m",
                advances_state=True,
                metadata_type="total_minutes",
                metadata_value=30.0,
                eval_only=True,
            ),
        ]
