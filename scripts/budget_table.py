# ABOUTME: Builds compute-budget vs SFT-fraction evaluation tables from a W&B project.
# ABOUTME: Reads pass@1 and test-time reward curves and renders markdown/CSV.

import argparse
import csv
import io
import json
import math
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# wandb pulls in pydantic models that attach `repr`/`frozen` to bare Field()
# calls, which pydantic flags with UnsupportedFieldAttributeWarning on import.
# The warnings come from a dependency we do not control and are pure noise for
# this reporting script, so silence just that message before wandb is imported.
warnings.filterwarnings(
    "ignore",
    message=r".*attribute with value .* was provided to the `Field\(\)` function.*",
)

from scripts.sweetspot_table import parse_wandb_url

BUDGETS_HOURS = [4, 16, 64]
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
MID_FRACTIONS = [0.25, 0.5, 0.75]
BENCHMARKS = ["gsm8k", "math500", "amc"]
TEST_TIME_REWARD_LABEL = "test-time reward"
TEST_TIME_REWARD_KEY = "eval/reward"

# A GRPO run whose SFT offset is below this counts as trained from base (0% column).
BASE_OFFSET_MINUTES = 30.0

# How far past a budget endpoint the forced eval is still accepted as that endpoint.
DEFAULT_TOLERANCE_MINUTES = 30.0

# A dead run is marked x if its furthest compute, or its last eval, lands within these
# of a budget: it reached the boundary but never logged the endpoint eval.
DEFAULT_NEAR_MINUTES = 25.0
DEFAULT_LAST_EVAL_NEAR_MINUTES = 30.0

TOTAL_MINUTES_KEY = "train/total_minutes"

# History rows per request; large pages cut round trips on the small eval scans.
DEFAULT_PAGE_SIZE = 10000

# Each run's history scan blocks on the network, so fetch several at once.
DEFAULT_WORKERS = 16

STATUS_VALUE = "value"
STATUS_RUNNING = "running"
STATUS_CRASHED_NEAR = "crashed_near"
STATUS_RANK = {STATUS_VALUE: 3, STATUS_RUNNING: 2, STATUS_CRASHED_NEAR: 1}
STATUS_MARKER = {STATUS_RUNNING: "o", STATUS_CRASHED_NEAR: "x"}

# Pseudo-benchmark holding the mean pass@1 across BENCHMARKS at each grid cell.
AVERAGE_LABEL = "average"


@dataclass
class MetricSource:
    """W&B coordinates needed to recover a metric's raw per-question samples."""

    global_step: int | None = None


@dataclass
class EvalEvent:
    """One evaluation cluster at a given total-compute point (minutes)."""

    total_minutes: float
    values: dict
    sources: dict = field(default_factory=dict)


@dataclass
class BudgetRun:
    """A W&B run reduced to what the budget table needs."""

    name: str
    tags: list
    state: str
    created_at: str
    offset_minutes: float
    events: list
    max_minutes: float = 0.0
    run_id: str = ""


@dataclass
class Cell:
    """A resolved table entry: a reading, or a status marker (o / x)."""

    benchmark: str
    budget_hours: int
    fraction: float
    value: float | None
    status: str = STATUS_VALUE
    run_name: str = ""
    run_id: str = ""
    standard_deviation: float | None = None
    variance_source: MetricSource | None = None


def is_sft_run(tags) -> bool:
    """Pure-SFT run: the 100% column."""
    return "sft" in tags and "grpo" not in tags


def is_grpo_run(tags) -> bool:
    """Any run that did GRPO (from base or from an SFT fork)."""
    return True
    return "grpo" in tags


def match_fraction_budget(
    offset_minutes: float, budgets=BUDGETS_HOURS, fractions=MID_FRACTIONS,
    tol_hours: float = 0.1,
) -> tuple | None:
    """Map an SFT compute offset to its (fraction, budget) grid cell.

    A fork that spent ``offset`` minutes on SFT sits at fraction*budget = offset,
    and fraction*budget is unique across the grid, so the offset identifies the cell.
    """
    offset_hours = offset_minutes / 60.0
    best = None
    best_err = tol_hours
    for budget in budgets:
        for fraction in fractions:
            err = abs(fraction * budget - offset_hours)
            if err < best_err:
                best_err = err
                best = (fraction, budget)
    return best


def read_after_endpoint(
    series, endpoint_minutes, tolerance_minutes: float = DEFAULT_TOLERANCE_MINUTES,
    epsilon_minutes: float = 1.0,
) -> float | None:
    """Value of the forced eval right at/after ``endpoint_minutes``.

    The training callback forces an eval on the first step whose total compute
    passes a budget, so the reading is the first sample at or just beyond the
    endpoint. Returns None if the run never reached the endpoint (or the sample
    lands beyond the tolerance window).
    """
    for total_minutes, value in series:
        if total_minutes >= endpoint_minutes - epsilon_minutes:
            if total_minutes <= endpoint_minutes + tolerance_minutes:
                return value
            return None
    return None


def read_max_through_endpoint(
    series, endpoint_minutes, tolerance_minutes: float = DEFAULT_TOLERANCE_MINUTES,
    epsilon_minutes: float = 1.0,
) -> float | None:
    """Maximum value seen through the forced eval at the endpoint.

    Like read_after_endpoint, this only returns a value once the run has logged
    an evaluation at the budget boundary (within the tolerance window). Earlier
    evaluations contribute to the maximum; evaluations after the accepted
    boundary evaluation do not.
    """
    values = []
    for total_minutes, value in series:
        values.append(value)
        if total_minutes >= endpoint_minutes - epsilon_minutes:
            if total_minutes <= endpoint_minutes + tolerance_minutes:
                return max(values)
            return None
    return None


def _series_for(run: BudgetRun, benchmark: str):
    """Sorted (total_minutes, value) samples for one benchmark."""
    return [
        (e.total_minutes, e.values[benchmark])
        for e in run.events
        if benchmark in e.values
    ]


def _event_after_endpoint(
    run: BudgetRun, benchmark: str, endpoint_minutes: float,
    tolerance_minutes: float = DEFAULT_TOLERANCE_MINUTES,
    epsilon_minutes: float = 1.0,
) -> EvalEvent | None:
    """The metric event at the accepted budget boundary."""
    for event in run.events:
        if benchmark not in event.values:
            continue
        if event.total_minutes >= endpoint_minutes - epsilon_minutes:
            if event.total_minutes <= endpoint_minutes + tolerance_minutes:
                return event
            return None
    return None


def _max_event_through_endpoint(
    run: BudgetRun, benchmark: str, endpoint_minutes: float,
    tolerance_minutes: float = DEFAULT_TOLERANCE_MINUTES,
    epsilon_minutes: float = 1.0,
) -> EvalEvent | None:
    """The event holding the maximum metric through an accepted boundary eval."""
    candidates = []
    for event in run.events:
        if benchmark not in event.values:
            continue
        candidates.append(event)
        if event.total_minutes >= endpoint_minutes - epsilon_minutes:
            if event.total_minutes <= endpoint_minutes + tolerance_minutes:
                return max(candidates, key=lambda item: item.values[benchmark])
            return None
    return None


def _crashed_at_boundary(
    run: BudgetRun, endpoint: float, near_minutes: float,
    last_eval_near_minutes: float,
) -> bool:
    """A dead run that reached the budget but never logged its endpoint eval."""
    if run.max_minutes >= endpoint - near_minutes:
        return True
    last_eval = run.events[-1].total_minutes if run.events else None
    return last_eval is not None and last_eval >= endpoint - last_eval_near_minutes


def assign_cells(
    run: BudgetRun, budgets=BUDGETS_HOURS,
    tolerance_minutes: float = DEFAULT_TOLERANCE_MINUTES,
    near_minutes: float = DEFAULT_NEAR_MINUTES,
    last_eval_near_minutes: float = DEFAULT_LAST_EVAL_NEAR_MINUTES,
    max_pass: bool = False,
) -> list:
    """All table cells a single run contributes (readings or o / x markers)."""
    cells = []
    if is_sft_run(run.tags):
        placements = [(b, 1.0) for b in budgets]
    elif is_grpo_run(run.tags):
        if run.offset_minutes < BASE_OFFSET_MINUTES:
            placements = [(b, 0.0) for b in budgets]
        else:
            match = match_fraction_budget(run.offset_minutes, budgets=budgets)
            placements = [(match[1], match[0])] if match else []
    else:
        placements = []

    is_live = run.state == "running"
    for budget_hours, fraction in placements:
        endpoint = budget_hours * 60.0
        crashed_near = not is_live and _crashed_at_boundary(
            run, endpoint, near_minutes, last_eval_near_minutes
        )
        metric_labels = list(BENCHMARKS)
        if not is_sft_run(run.tags):
            metric_labels.append(TEST_TIME_REWARD_LABEL)
        for benchmark in metric_labels:
            event_reader = _max_event_through_endpoint if max_pass else _event_after_endpoint
            event = event_reader(
                run, benchmark, endpoint,
                tolerance_minutes=tolerance_minutes,
            )
            value = event.values[benchmark] if event is not None else None
            if value is not None:
                status = STATUS_VALUE
            elif is_live:
                status = STATUS_RUNNING
            elif crashed_near:
                status = STATUS_CRASHED_NEAR
            else:
                continue
            cells.append(
                Cell(
                    benchmark, budget_hours, fraction, value, status,
                    run.name, run.run_id,
                    variance_source=(event.sources.get(benchmark) if event else None),
                )
            )
    return cells


_STATE_RANK = {"finished": 3, "running": 2, "crashed": 1}


def build_grid(
    runs, budgets=BUDGETS_HOURS,
    tolerance_minutes: float = DEFAULT_TOLERANCE_MINUTES,
    near_minutes: float = DEFAULT_NEAR_MINUTES,
    last_eval_near_minutes: float = DEFAULT_LAST_EVAL_NEAR_MINUTES,
    max_pass: bool = False,
) -> dict:
    """Resolve runs into a {(benchmark, budget, fraction): Cell} grid.

    A real reading beats a running marker beats a crashed-at-boundary marker; for
    equal status, prefer finished over running over crashed, then most recent.
    """
    ordered = sorted(runs, key=lambda r: (_STATE_RANK.get(r.state, 0), r.created_at))
    grid = {}
    for run in ordered:
        for cell in assign_cells(
            run, budgets=budgets, tolerance_minutes=tolerance_minutes,
            near_minutes=near_minutes, last_eval_near_minutes=last_eval_near_minutes,
            max_pass=max_pass,
        ):
            key = (cell.benchmark, cell.budget_hours, cell.fraction)
            current = grid.get(key)
            if current is None or STATUS_RANK[cell.status] >= STATUS_RANK[current.status]:
                grid[key] = cell
    return grid


def pass_at_1_standard_deviation(table: dict) -> float:
    """Plug-in SD of mean pass@1 from each question's repeated responses.

    For question i with m_i responses and empirical success rate p_i, this uses
    Var(p_i) = p_i(1-p_i)/m_i and the independent-question sum-variance rule.
    """
    columns = table.get("columns", [])
    try:
        correct_index = columns.index("per_response_correct")
    except ValueError as exc:
        raise ValueError("pass@1 table lacks per_response_correct") from exc

    question_variances = []
    for row in table.get("data", []):
        values = row[correct_index]
        if isinstance(values, str):
            values = json.loads(values)
        if not isinstance(values, (list, tuple)) or not values:
            continue
        numeric = [float(value) for value in values]
        sample_count = len(numeric)
        probability = sum(numeric) / sample_count
        question_variances.append(
            probability * (1.0 - probability) / sample_count
        )

    question_count = len(question_variances)
    if question_count == 0:
        raise ValueError("pass@1 table has no usable per-question responses")
    variance = sum(question_variances) / (question_count ** 2)
    return math.sqrt(max(variance, 0.0))


def _table_ref_path(table_ref) -> str:
    if isinstance(table_ref, str):
        return table_ref
    get_value = getattr(table_ref, "get", None)
    if get_value is not None:
        path = get_value("path")
        if path:
            return path
    raise ValueError("W&B table reference has no downloadable path")


def _load_wandb_table(run, table_ref) -> dict:
    """Download one existing W&B table without logging any new metrics."""
    remote_path = _table_ref_path(table_ref)
    with tempfile.TemporaryDirectory(prefix="budget-table-") as temp_dir:
        downloaded = run.file(remote_path).download(root=temp_dir, replace=True)
        local_path = Path(getattr(downloaded, "name", downloaded))
        with local_path.open(encoding="utf-8") as handle:
            return json.load(handle)


def populate_standard_deviations(grid, raw_runs) -> list[str]:
    """Populate pass@1 benchmark cells from existing raw W&B tables; return warnings.

    Only the events that actually won a grid cell are downloaded. In --max mode,
    the variance therefore belongs to the same historical event as the maximum.
    The test-time reward metric carries no SD: its retained completion tables do
    not cover the full eval, so only its mean is reported.
    """
    runs_by_id = {run.id: run for run in raw_runs}
    cache = {}
    warnings = []

    for cell in grid.values():
        if cell.status != STATUS_VALUE or cell.variance_source is None:
            continue
        if cell.benchmark not in BENCHMARKS:
            continue
        run = runs_by_id.get(cell.run_id)
        if run is None:
            continue
        source = cell.variance_source
        try:
            if source.global_step is None:
                raise ValueError("metric history has no train/global_step")
            table_key = f"raw_generations/{cell.benchmark}/step_{source.global_step}"
            table_ref = run.summary.get(table_key)
            if table_ref is None:
                raise ValueError(f"run summary has no {table_key}")

            cache_key = (cell.run_id, _table_ref_path(table_ref), cell.benchmark)
            result = cache.get(cache_key)
            if result is None:
                result = pass_at_1_standard_deviation(_load_wandb_table(run, table_ref))
                cache[cache_key] = result
            cell.standard_deviation = result
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(
                f"{cell.run_name} {cell.benchmark} {cell.budget_hours}h "
                f"{int(cell.fraction * 100)}% SFT: {exc}"
            )
    return warnings


def add_average_cells(
    grid, benchmarks=BENCHMARKS, budgets=BUDGETS_HOURS, fractions=FRACTIONS,
) -> None:
    """Add ("average", budget, fraction) cells: the mean benchmark reading per cell.

    Averages the STATUS_VALUE readings across benchmarks (they are logged together,
    so a cell usually has all or none). With no readings, the cell carries the
    best-ranked o / x marker so the average table lines up with the per-benchmark ones.
    """
    for budget in budgets:
        for fraction in fractions:
            cells = [grid.get((b, budget, fraction)) for b in benchmarks]
            cells = [c for c in cells if c is not None]
            if not cells:
                continue
            marker = max(cells, key=lambda c: STATUS_RANK[c.status])
            readings = [c for c in cells if c.status == STATUS_VALUE]
            if readings:
                value = sum(c.value for c in readings) / len(readings)
                status = STATUS_VALUE
                if all(c.standard_deviation is not None for c in readings):
                    standard_deviation = math.sqrt(sum(
                        c.standard_deviation ** 2 for c in readings
                    )) / len(readings)
                else:
                    standard_deviation = None
            else:
                value = None
                status = marker.status
                standard_deviation = None
            grid[(AVERAGE_LABEL, budget, fraction)] = Cell(
                AVERAGE_LABEL, budget, fraction, value, status,
                marker.run_name, marker.run_id,
                standard_deviation=standard_deviation,
            )


def _fmt_frac_header(fraction: float) -> str:
    return f"{int(round(fraction * 100))}%"


def format_table(grid, benchmark: str, budgets=BUDGETS_HOURS, fractions=FRACTIONS) -> str:
    """Render one benchmark's budget table as GitHub-flavored markdown."""
    header = ["% of compute for SFT $\\rightarrow$"] + [_fmt_frac_header(f) for f in fractions]
    sep = ["---"] + [":--:" for _ in fractions]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        "| Total Compute Hours $\\downarrow$ |" + " |" * len(fractions),
    ]
    for budget in budgets:
        row = [f"{budget} Hours"]
        for fraction in fractions:
            row.append(_render_cell(grid.get((benchmark, budget, fraction))))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def format_standard_deviation_table(
    grid, benchmark: str, budgets=BUDGETS_HOURS, fractions=FRACTIONS,
) -> str:
    """Render the SD of each reported mean as a markdown table."""
    header = ["% of compute for SFT $\\rightarrow$"] + [
        _fmt_frac_header(f) for f in fractions
    ]
    sep = ["---"] + [":--:" for _ in fractions]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        "| Total Compute Hours $\\downarrow$ |" + " |" * len(fractions),
    ]
    for budget in budgets:
        row = [f"{budget} Hours"]
        for fraction in fractions:
            row.append(_render_standard_deviation_cell(
                grid.get((benchmark, budget, fraction))
            ))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def format_id_table(
    grid, benchmark: str, budgets=BUDGETS_HOURS, fractions=FRACTIONS,
) -> str:
    """Render the W&B run ID behind each cell as a markdown table."""
    header = ["% of compute for SFT $\\rightarrow$"] + [_fmt_frac_header(f) for f in fractions]
    sep = ["---"] + [":--:" for _ in fractions]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        "| Total Compute Hours $\\downarrow$ |" + " |" * len(fractions),
    ]
    for budget in budgets:
        row = [f"{budget} Hours"]
        for fraction in fractions:
            row.append(_render_run_id(grid.get((benchmark, budget, fraction))))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _render_cell(cell) -> str:
    """A percentage for a reading, else the o / x marker (blank if no run)."""
    if cell is None:
        return ""
    if cell.status == STATUS_VALUE:
        return f"{cell.value * 100:.1f}%"
    return STATUS_MARKER.get(cell.status, "")


def _render_standard_deviation_cell(cell) -> str:
    """SD percentage for a reading, else the o / x marker (blank if unknown)."""
    if cell is None:
        return ""
    if cell.status != STATUS_VALUE:
        return STATUS_MARKER.get(cell.status, "")
    if cell.standard_deviation is None:
        return ""
    return f"{cell.standard_deviation * 100:.1f}%"


def _render_combined_cell(cell) -> str:
    """A reading as ``mean ± SD``, else the o / x marker (blank if no run).

    The ``± SD`` is appended only when the standard deviation was recovered
    (pass@1 benchmarks); a bare mean is shown otherwise so the cell still
    reports its value instead of collapsing to nothing.
    """
    if cell is None:
        return ""
    if cell.status != STATUS_VALUE:
        return STATUS_MARKER.get(cell.status, "")
    rendered = f"{cell.value * 100:.1f}%"
    if cell.standard_deviation is not None:
        rendered += f" ± {cell.standard_deviation * 100:.1f}%"
    return rendered


def _render_run_id(cell) -> str:
    """The W&B ID for a populated cell (blank if no run fills it)."""
    return cell.run_id if cell is not None else ""


def format_csv(grid, benchmark: str, budgets=BUDGETS_HOURS, fractions=FRACTIONS) -> str:
    """Render one benchmark's budget table as CSV a spreadsheet can import.

    Same cells as format_table, without the markdown/LaTeX; the "60.4%" strings
    import as percentages. Rows are budgets, columns are SFT fractions.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Total Compute Hours"] + [_fmt_frac_header(f) for f in fractions])
    for budget in budgets:
        row = [f"{budget} Hours"]
        for fraction in fractions:
            row.append(_render_cell(grid.get((benchmark, budget, fraction))))
        writer.writerow(row)
    return buffer.getvalue()


def format_standard_deviation_csv(
    grid, benchmark: str, budgets=BUDGETS_HOURS, fractions=FRACTIONS,
) -> str:
    """Render standard deviations as CSV (blank where no SD was recovered)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Total Compute Hours"] + [_fmt_frac_header(f) for f in fractions])
    for budget in budgets:
        row = [f"{budget} Hours"]
        for fraction in fractions:
            row.append(_render_standard_deviation_cell(
                grid.get((benchmark, budget, fraction))
            ))
        writer.writerow(row)
    return buffer.getvalue()


def format_combined_table(
    grid, benchmark: str, budgets=BUDGETS_HOURS, fractions=FRACTIONS,
) -> str:
    """Render one benchmark's ``mean ± SD`` budget table as markdown."""
    header = ["% of compute for SFT $\\rightarrow$"] + [
        _fmt_frac_header(f) for f in fractions
    ]
    sep = ["---"] + [":--:" for _ in fractions]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        "| Total Compute Hours $\\downarrow$ |" + " |" * len(fractions),
    ]
    for budget in budgets:
        row = [f"{budget} Hours"]
        for fraction in fractions:
            row.append(_render_combined_cell(
                grid.get((benchmark, budget, fraction))
            ))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def format_combined_csv(
    grid, benchmark: str, budgets=BUDGETS_HOURS, fractions=FRACTIONS,
) -> str:
    """Render ``mean ± SD`` as CSV (only pass@1 benchmarks carry the ± term)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Total Compute Hours"] + [_fmt_frac_header(f) for f in fractions])
    for budget in budgets:
        row = [f"{budget} Hours"]
        for fraction in fractions:
            row.append(_render_combined_cell(
                grid.get((benchmark, budget, fraction))
            ))
        writer.writerow(row)
    return buffer.getvalue()


def format_id_csv(
    grid, benchmark: str, budgets=BUDGETS_HOURS, fractions=FRACTIONS,
) -> str:
    """Render the W&B run ID behind each cell as CSV."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Total Compute Hours"] + [_fmt_frac_header(f) for f in fractions])
    for budget in budgets:
        row = [f"{budget} Hours"]
        for fraction in fractions:
            row.append(_render_run_id(grid.get((benchmark, budget, fraction))))
        writer.writerow(row)
    return buffer.getvalue()


def filter_runs_by_tag(runs, tag):
    """Keep only runs carrying ``tag`` — scopes the grid to one attempt (e.g. try3)."""
    if not tag:
        return list(runs)
    return [r for r in runs if tag in r.tags]


def _fetch_events(run, page_size: int = DEFAULT_PAGE_SIZE) -> tuple[list, float]:
    """Pull eval events and the SFT offset from a run's history.

    Each metric is logged on a step that also carries total_minutes, so one keyed
    scan per metric returns just those points already stamped with total compute.
    Requesting all keys at once would return nothing because the metrics can live
    on separate rows. Events sharing a total_minutes merge into one EvalEvent.

    The offset is the earliest pass@1 evaluation, which identifies the SFT compute
    a fork resumed from. Reward-only points do not affect run placement.
    """
    by_minutes = {}
    pass_minutes = []
    history_keys = {
        **{benchmark: f"eval/{benchmark}_pass_at_1" for benchmark in BENCHMARKS},
        TEST_TIME_REWARD_LABEL: TEST_TIME_REWARD_KEY,
    }
    for metric_label, key in history_keys.items():
        keys = [TOTAL_MINUTES_KEY, key, "train/global_step"]
        for row in run.scan_history(keys=keys, page_size=page_size):
            tm = row.get(TOTAL_MINUTES_KEY)
            value = row.get(key)
            if tm is None or value is None:
                continue
            cluster = next((t for t in by_minutes if abs(t - tm) <= 1e-06), tm)
            event_data = by_minutes.setdefault(
                cluster, {"values": {}, "sources": {}},
            )
            event_data["values"][metric_label] = value
            event_data["sources"][metric_label] = MetricSource(
                global_step=row.get("train/global_step"),
            )
            if metric_label in BENCHMARKS:
                pass_minutes.append(tm)

    events = [
        EvalEvent(
            total_minutes=tm,
            values=event_data["values"],
            sources=event_data["sources"],
        )
        for tm, event_data in sorted(by_minutes.items())
    ]
    offset = min(pass_minutes) if pass_minutes else 0.0
    return events, offset


def _wandb_run_to_budget_run(run, page_size: int = DEFAULT_PAGE_SIZE) -> BudgetRun:
    events, offset = _fetch_events(run, page_size=page_size)
    max_minutes = run.summary.get(TOTAL_MINUTES_KEY) or 0.0
    return BudgetRun(
        name=run.name,
        tags=list(run.tags),
        state=run.state,
        created_at=run.created_at,
        offset_minutes=offset,
        events=events,
        max_minutes=max_minutes,
        run_id=run.id,
    )


def fetch_budget_runs(
    raw_runs, workers: int = DEFAULT_WORKERS, page_size: int = DEFAULT_PAGE_SIZE,
) -> list:
    """Convert W&B runs to BudgetRuns, reading their histories concurrently.

    Each run's history scan blocks on the network, so a thread pool overlaps the
    per-run latencies. Order is not preserved; build_grid sorts by state/time.
    """
    if workers <= 1:
        return [_wandb_run_to_budget_run(r, page_size=page_size) for r in raw_runs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda r: _wandb_run_to_budget_run(r, page_size=page_size), raw_runs))


def main(argv: list[str] | None = None) -> None:
    """CLI: print budget tables for each benchmark from a W&B project URL."""
    import wandb

    parser = argparse.ArgumentParser(
        description="Compute-budget vs SFT-fraction evaluation tables from W&B"
    )
    parser.add_argument("url", help="W&B project URL")
    parser.add_argument(
        "--tolerance-minutes", type=float, default=DEFAULT_TOLERANCE_MINUTES,
        help="How far past a budget the forced eval may land (default: 30)",
    )
    parser.add_argument(
        "--near-minutes", type=float, default=DEFAULT_NEAR_MINUTES,
        help="A dead run stopping within this of a budget is marked x (default: 25)",
    )
    parser.add_argument(
        "--last-eval-near-minutes", type=float, default=DEFAULT_LAST_EVAL_NEAR_MINUTES,
        help="A dead run whose last eval is within this of a budget is marked x (default: 30)",
    )
    parser.add_argument(
        "--tag", default="try3",
        help="Only use runs carrying this tag (e.g. try3), to scope to one attempt",
    )
    parser.add_argument(
        "--raw", action="store_true", help="Also print which run fills each cell"
    )
    parser.add_argument(
        "--max", dest="max_pass", action="store_true",
        help="Use the maximum metric value seen through each time budget",
    )
    parser.add_argument(
        "--csv", action="store_true", default=True,
        help="Emit CSV (importable into Google Sheets) instead of markdown",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Runs whose history is fetched in parallel (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--page-size", type=int, default=DEFAULT_PAGE_SIZE,
        help=f"History rows per W&B request (default: {DEFAULT_PAGE_SIZE})",
    )
    args = parser.parse_args(argv)

    entity, project = parse_wandb_url(args.url)
    api = wandb.Api()
    raw_runs = filter_runs_by_tag(list(api.runs(f"{entity}/{project}")), args.tag)
    if args.tag and not raw_runs:
        print(f"No runs tagged '{args.tag}' in {entity}/{project}", file=sys.stderr)
        sys.exit(1)
    runs = fetch_budget_runs(raw_runs, workers=args.workers, page_size=args.page_size)
    grid = build_grid(
        runs, tolerance_minutes=args.tolerance_minutes,
        near_minutes=args.near_minutes,
        last_eval_near_minutes=args.last_eval_near_minutes,
        max_pass=args.max_pass,
    )
    variance_warnings = populate_standard_deviations(grid, raw_runs)
    add_average_cells(grid)

    legend = "o = run still training toward this budget; x = run died at this budget before its eval (re-evaluate)"

    print(
        legend if args.csv else f"_{legend}_\n",
        file=sys.stderr if args.csv else sys.stdout,
    )
    for warning in variance_warnings:
        print(f"standard-deviation warning: {warning}", file=sys.stderr)
    print(format_id_csv(grid, AVERAGE_LABEL))
    for benchmark in BENCHMARKS + [AVERAGE_LABEL, TEST_TIME_REWARD_LABEL]:
        if args.csv:
            print(f"{project} — {benchmark}")
            print(format_combined_csv(grid, benchmark))
        else:
            print(f"### {project} — {benchmark}\n")
            print(format_combined_table(grid, benchmark))
        if args.raw:
            for budget in BUDGETS_HOURS:
                for fraction in FRACTIONS:
                    cell = grid.get((benchmark, budget, fraction))
                    if cell:
                        shown = (
                            f"{cell.value * 100:6.2f}%"
                            if cell.status == STATUS_VALUE
                            else f"{STATUS_MARKER.get(cell.status, '?'):>7}"
                        )
                        print(
                            f"  {budget}h {int(fraction * 100):>3}%  {shown}  <- {cell.run_name}",
                            file=sys.stderr,
                        )
            print("", file=sys.stderr)


if __name__ == "__main__":
    main()
