# ABOUTME: Generates a markdown sweetspot analysis table from a W&B project URL.
# ABOUTME: Shows SFT→DPO accuracy trade-offs for the most recent experiment run.

from dataclasses import dataclass, field
from urllib.parse import urlparse


def parse_wandb_url(url: str) -> tuple[str, str]:
    """Extract (entity, project) from a W&B project URL."""
    parsed = urlparse(url)
    if parsed.hostname != "wandb.ai":
        raise ValueError(f"Not a wandb.ai URL: {url}")
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"URL must contain entity/project: {url}")
    return parts[0], parts[1]


@dataclass
class RunInfo:
    """Lightweight representation of a W&B run for sweetspot analysis."""

    name: str
    tags: list[str]
    created_at: str
    state: str = "finished"
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def _group_tags(tags: list[str]) -> set[str]:
    """Extract the group-identifying tags (not sft/dpo, not numeric, not early_pair:*)."""
    group = set()
    for t in tags:
        if t in ("sft", "dpo"):
            continue
        if t.startswith("early_pair:"):
            continue
        try:
            float(t)
            continue
        except ValueError:
            pass
        try:
            int(t)
            continue
        except ValueError:
            pass
        group.add(t)
    return group


def find_latest_sft_run(runs: list[RunInfo]) -> RunInfo:
    """Find the most recently created finished SFT run."""
    sft_runs = [
        r for r in runs if "sft" in r.tags and r.state == "finished"
    ]
    if not sft_runs:
        raise ValueError("No finished SFT runs found")
    sft_runs.sort(key=lambda r: r.created_at, reverse=True)
    return sft_runs[0]


def find_dpo_forks(sft_run: RunInfo, runs: list[RunInfo]) -> list[RunInfo]:
    """Find finished DPO runs that belong to the same experiment group as the SFT run."""
    sft_group = _group_tags(sft_run.tags)
    forks = []
    for r in runs:
        if "dpo" not in r.tags:
            continue
        if r.state != "finished":
            continue
        if r.created_at <= sft_run.created_at:
            continue
        if _group_tags(r.tags) != sft_group:
            continue
        forks.append(r)
    return forks


@dataclass
class SftBaseline:
    """Baseline metrics from the full SFT run."""

    total_budget: int
    base_acc: float | None
    sft_full_acc: float


@dataclass
class DpoRow:
    """One row of the sweetspot table, representing a single DPO fork."""

    pass_at_1_threshold: float
    sft_data: int
    dpo_data: int
    sft_acc_at_stop: float
    best_dpo_acc: float
    final_dpo_acc: float


def extract_sft_baseline(
    sft_run: RunInfo, history: list[float]
) -> SftBaseline:
    """Extract baseline metrics from the SFT run."""
    cfg = sft_run.config
    steps = sft_run.summary["train/global_step"]
    batch = cfg["per_device_train_batch_size"]
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    epochs = cfg["num_train_epochs"]
    total_budget = int(steps * batch * grad_accum / epochs)

    sft_full_acc = sft_run.summary["eval/gsm8k_pass_at_1"]
    base_acc = history[0] if history else None

    return SftBaseline(
        total_budget=total_budget,
        base_acc=base_acc,
        sft_full_acc=sft_full_acc,
    )


def _parse_numeric_tags(tags: list[str]) -> tuple[float | None, int | None]:
    """Extract the pass@1 threshold (float) and SFT data count (int) from tags."""
    threshold = None
    sft_data = None
    for t in tags:
        if t in ("sft", "dpo") or t.startswith("early_pair:"):
            continue
        try:
            val = float(t)
            if val < 1.0:
                threshold = val
            else:
                sft_data = int(t)
        except ValueError:
            continue
    return threshold, sft_data


def extract_dpo_row(
    dpo_run: RunInfo, total_budget: int, history: list[float]
) -> DpoRow:
    """Extract a sweetspot table row from a DPO run."""
    threshold, sft_data = _parse_numeric_tags(dpo_run.tags)
    dpo_data = total_budget - sft_data if sft_data is not None else 0

    return DpoRow(
        pass_at_1_threshold=threshold or 0.0,
        sft_data=sft_data or 0,
        dpo_data=dpo_data,
        sft_acc_at_stop=history[0],
        best_dpo_acc=max(history),
        final_dpo_acc=history[-1],
    )


def _fmt_pct(val: float) -> str:
    """Format a ratio as a percentage string (e.g. 0.5110 -> '51.10%')."""
    return f"{val * 100:.2f}%"


def _fmt_delta(val: float, ref: float) -> str:
    """Format the difference between two ratios as percentage points."""
    delta = (val - ref) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}pp"


def format_table(baseline: SftBaseline, rows: list[DpoRow]) -> str:
    """Render the sweetspot analysis as a markdown table."""
    include_base = baseline.base_acc is not None
    sorted_rows = sorted(rows, key=lambda r: r.sft_data, reverse=True)

    headers = [
        "Pass@1 Threshold", "SFT Data", "DPO Data",
        "SFT Acc at Stop", "Best DPO Acc", "Final DPO Acc", "vs SFT-full",
    ]
    if include_base:
        headers.append("vs SFT-0")

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    # SFT-only baseline row
    sft_cells = [
        "SFT-only",
        str(baseline.total_budget),
        "0",
        _fmt_pct(baseline.sft_full_acc),
        _fmt_pct(baseline.sft_full_acc),
        _fmt_pct(baseline.sft_full_acc),
        "baseline",
    ]
    if include_base:
        assert baseline.base_acc is not None
        sft_cells.append(_fmt_delta(baseline.sft_full_acc, baseline.base_acc))
    lines.append("| " + " | ".join(sft_cells) + " |")

    # DPO fork rows
    for row in sorted_rows:
        cells = [
            f"{row.pass_at_1_threshold:.2f}",
            str(row.sft_data),
            str(row.dpo_data),
            _fmt_pct(row.sft_acc_at_stop),
            _fmt_pct(row.best_dpo_acc),
            _fmt_pct(row.final_dpo_acc),
            _fmt_delta(row.best_dpo_acc, baseline.sft_full_acc),
        ]
        if include_base:
            assert baseline.base_acc is not None
            cells.append(_fmt_delta(row.best_dpo_acc, baseline.base_acc))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"
