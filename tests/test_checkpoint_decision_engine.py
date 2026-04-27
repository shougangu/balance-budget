# ABOUTME: Unit tests for CheckpointDecisionEngine — pure threshold/early-tuple/gap logic.
# ABOUTME: No vllm or wandb mocks needed (engine has zero such deps).

from tuning.training.passk.decisions import CheckpointDecision, CheckpointDecisionEngine


def _engine(thresholds=None, early_tuples=None, max_gap=None):
    return CheckpointDecisionEngine(
        target_thresholds=thresholds or [],
        early_tuples=early_tuples or None,
        max_checkpoint_gap=max_gap,
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
