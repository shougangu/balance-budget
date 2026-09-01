# ABOUTME: Draws the SFT/RL compute frontier per model: every eval point of the SFT run and
# ABOUTME: of each GRPO worker forked from it, on a FLOPs axis, with the Pareto envelope.

import argparse
import csv
import json
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*attribute with value .* was provided to the `Field\(\)` function.*",
)

WANDB_ENTITY = "shougan-university-of-waterloo"
TOTAL_MINUTES_KEY = "train/total_minutes"
GLOBAL_STEP_KEY = "train/global_step"
NUM_TOKENS_KEY = "train/num_tokens"
PAGE_SIZE = 10000
WORKERS = 16
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "budget_frontier"
TOKEN_LENGTHS_PATH = DEFAULT_OUT_DIR / "sft_token_lengths.json"

# Active non-embedding text parameters, the N of the 6NT / 10NT FLOP estimates
# (embeddings, lm_head and the Gemma vision tower excluded).
NON_EMBEDDING_PARAMS = {
    "llama3-3B": 2_818_747_392,
    "llama3-8B": 6_979_588_096,
    "gemma3-4B": 3_209_010_688,
    "gemma3-12B": 10_759_155_456,
    "qwen3-4B": 3_633_511_936,
    "qwen3-8B": 6_946_075_648,
    "qwen3-14B": 13_212_482_560,
}

# Runs that carry the campaign tag but are not part of the lineage.
EXCLUDED_TAGS = {"collator-glitch"}


@dataclass(frozen=True)
class Lineage:
    project: str
    tag: str
    model: str
    token_spec: str
    metrics: tuple
    title: str


LINEAGES = {
    "l3b": Lineage("[1]math-l3b", "redo", "llama3-3B", "llama3-3B|simple|sft-openmath|1024",
                   ("math500_pass_at_1", "math500_pass_at_4"), "Llama-3.2 3B / MATH-500"),
    "l8b": Lineage("[1]math-l8b", "redo", "llama3-8B", "llama3-8B|simple|sft-openmath|1024",
                   ("math500_pass_at_1", "math500_pass_at_4"), "Llama-3.1 8B / MATH-500"),
    "g4b": Lineage("[1]math-g4b", "redo", "gemma3-4B", "gemma3-4B|simple|sft-openmath|1024",
                   ("math500_pass_at_1", "math500_pass_at_4"), "Gemma-3 4B / MATH-500"),
    "g12b": Lineage("[1]math-g12b", "redo_evalBS", "gemma3-12B", "gemma3-12B|simple|sft-openmath|1024",
                    ("math500_pass_at_1", "math500_pass_at_4"), "Gemma-3 12B / MATH-500"),
    "g12b-if": Lineage("[1]if-g12b", "redo", "gemma3-12B", "gemma3-12B|gemma-3|sft-ifmix|4096",
                       ("pass_at_1", "ifbench_pass_at_1"), "Gemma-3 12B / IFEval"),
}

METRIC_LABELS = {
    "math500_pass_at_1": "MATH-500 pass@1",
    "math500_pass_at_4": "MATH-500 pass@4",
    "pass_at_1": "IFEval pass@1",
    "ifbench_pass_at_1": "IFBench pass@1",
}


@dataclass
class Point:
    """One evaluation of one checkpoint, placed on the compute axis."""
    run_id: str
    mark_minutes: float
    total_minutes: float
    sft_flops: float
    rl_flops: float
    value: float

    @property
    def total_flops(self):
        return self.sft_flops + self.rl_flops

    @property
    def rl_fraction(self):
        return self.rl_flops / self.total_flops if self.total_flops else 0.0


@dataclass
class Series:
    """The eval trajectory of one run, for one metric."""
    run_id: str
    mark_minutes: float
    points: list = field(default_factory=list)


def sft_flops(n_params, examples, mean_tokens):
    """6·N·T for the SFT stage, T = examples processed × mean tokens per example."""
    return 6 * n_params * examples * mean_tokens


def rl_flops(n_params, rollout_tokens, reference_model=True):
    """GRPO compute: one rollout forward (2N), one reference forward (2N) and the
    policy update (6N) per rollout token."""
    return (10 if reference_model else 8) * n_params * rollout_tokens


def parse_worker_tags(tags):
    """Return (SFT mark in minutes, SFT examples consumed at the fork) from a GRPO worker's
    tags, or None for runs that are not forks (the SFT run, RL-from-base)."""
    marks = [float(t) for t in tags if "." in t and t.replace(".", "", 1).isdigit()]
    examples = [int(t) for t in tags if t.isdigit()]
    if len(marks) != 1 or len(examples) != 1 or examples[0] == 0:
        return None
    return marks[0], examples[0]


def tokens_at_step(token_rows, step):
    """Cumulative rollout tokens at the last logged step at or before `step`."""
    tokens = 0
    for logged_step, num_tokens in token_rows:
        if logged_step > step:
            break
        tokens = num_tokens
    return tokens


def pareto_frontier(points):
    """Points that beat every point of equal or smaller compute, in compute order."""
    frontier = []
    best = -math.inf
    for point in sorted(points, key=lambda p: (p.total_flops, -p.value)):
        if point.value > best:
            frontier.append(point)
            best = point.value
    return frontier


def first_frontier_touch(frontier):
    """The first frontier point of each run, keyed by run id."""
    touches = {}
    for point in frontier:
        touches.setdefault(point.run_id, point)
    return touches


# --- W&B fetch ---------------------------------------------------------------


def _scan(run, keys):
    return list(run.scan_history(keys=keys, page_size=PAGE_SIZE))


def _fetch_run(run, metrics):
    """Raw history needed for one run: eval rows per metric and the token counter."""
    evals = {}
    for metric in metrics:
        key = f"eval/{metric}"
        evals[metric] = [
            {"step": row[GLOBAL_STEP_KEY], "total_minutes": row[TOTAL_MINUTES_KEY], "value": row[key]}
            for row in _scan(run, [TOTAL_MINUTES_KEY, key, GLOBAL_STEP_KEY])
            if row.get(key) is not None and row.get(TOTAL_MINUTES_KEY) is not None
        ]
    is_sft = "sft" in run.tags
    tokens = [] if is_sft else [
        [row[GLOBAL_STEP_KEY], row[NUM_TOKENS_KEY]]
        for row in _scan(run, [NUM_TOKENS_KEY, GLOBAL_STEP_KEY])
        if row.get(NUM_TOKENS_KEY) is not None
    ]
    config = run.config
    return {
        "id": run.id,
        "name": run.name,
        "tags": list(run.tags),
        "state": run.state,
        "is_sft": is_sft,
        "examples_per_step": (
            int(config.get("per_device_train_batch_size", 1))
            * int(config.get("gradient_accumulation_steps", 1))
        ),
        "beta": config.get("beta"),
        "evals": evals,
        "tokens": tokens,
    }


def fetch_lineage(api, lineage):
    runs = [
        r for r in api.runs(f"{WANDB_ENTITY}/{lineage.project}", filters={"tags": {"$in": [lineage.tag]}})
        if not EXCLUDED_TAGS & set(r.tags)
    ]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(lambda r: _fetch_run(r, lineage.metrics), runs))


def load_or_fetch(name, lineage, cache_dir, refresh):
    cache = cache_dir / f"{name}_wandb.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    import wandb
    raw = fetch_lineage(wandb.Api(timeout=60), lineage)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(raw))
    return raw


# --- Points ------------------------------------------------------------------


def build_series(raw_runs, metric, n_params, mean_tokens):
    """Place every eval of every run on the FLOPs axis for one metric."""
    series = []
    for run in raw_runs:
        evals = run["evals"].get(metric, [])
        if not evals:
            continue
        if run["is_sft"]:
            s = Series(run["id"], mark_minutes=0.0)
            for ev in evals:
                examples = ev["step"] * run["examples_per_step"]
                s.points.append(Point(
                    run["id"], 0.0, ev["total_minutes"],
                    sft_flops(n_params, examples, mean_tokens), 0.0, ev["value"],
                ))
        else:
            parsed = parse_worker_tags(run["tags"])
            if parsed is None:
                continue
            mark, examples = parsed
            fork_sft = sft_flops(n_params, examples, mean_tokens)
            reference_model = (run["beta"] or 0.0) > 0
            s = Series(run["id"], mark_minutes=mark)
            for ev in evals:
                rollout = tokens_at_step(run["tokens"], ev["step"])
                s.points.append(Point(
                    run["id"], mark, ev["total_minutes"], fork_sft,
                    rl_flops(n_params, rollout, reference_model), ev["value"],
                ))
        # The pre-SFT base-model eval has zero compute and no place on a log axis.
        s.points = sorted((p for p in s.points if p.total_flops > 0), key=lambda p: p.total_flops)
        if s.points:
            series.append(s)
    return sorted(series, key=lambda s: s.mark_minutes)


def write_points_csv(path, name, metric, series):
    with path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        for s in series:
            for p in s.points:
                writer.writerow([
                    name, metric, s.run_id, f"{p.mark_minutes:g}", f"{p.total_minutes:.2f}",
                    f"{p.sft_flops:.4e}", f"{p.rl_flops:.4e}", f"{p.total_flops:.4e}",
                    f"{p.rl_fraction:.4f}", f"{p.value:.4f}",
                ])


# --- Figure ------------------------------------------------------------------


def _mark_color(cmap, mark, marks):
    lo, hi = math.log(min(marks)), math.log(max(marks))
    t = 0.0 if hi == lo else (math.log(mark) - lo) / (hi - lo)
    return cmap(0.35 + 0.65 * t)


def draw_panel(ax, series, label_extremes=False):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("Blues")
    workers = [s for s in series if s.mark_minutes > 0]
    marks = sorted({s.mark_minutes for s in workers}) or [1.0]
    all_points = [p for s in series for p in s.points]

    for s in series:
        xs = [p.total_flops for p in s.points]
        ys = [100 * p.value for p in s.points]
        if s.mark_minutes == 0:
            ax.plot(xs, ys, color="#8a8a8a", lw=1.5, ls="--", zorder=2, label="SFT only")
            ax.scatter(xs, ys, s=28, facecolors="white", edgecolors="#8a8a8a", lw=1.2, zorder=3)
            continue
        color = _mark_color(cmap, s.mark_minutes, marks)
        ax.plot(xs, ys, color=color, lw=1.5, zorder=4)
        ax.scatter(xs[1:], ys[1:], s=22, color=color, edgecolors="white", lw=0.8, zorder=5)
        ax.scatter(xs[:1], ys[:1], s=40, facecolors="white", edgecolors=color, lw=1.6, zorder=6)

    frontier = pareto_frontier(all_points)
    ax.step(
        [p.total_flops for p in frontier], [100 * p.value for p in frontier],
        where="post", color="black", lw=1.6, zorder=7, label="Pareto frontier",
    )
    touches = first_frontier_touch(frontier)
    labelled = set(marks if label_extremes else marks[1:-1])
    for s in workers:
        point = touches.get(s.run_id)
        if point is None or s.mark_minutes not in labelled or point.rl_fraction == 0:
            continue
        ax.annotate(
            f"RL {100 * point.rl_fraction:.0f}%", (point.total_flops, 100 * point.value),
            xytext=(4, -11), textcoords="offset points", fontsize=7, color="#222222", zorder=8,
        )
    ax.set_xscale("log")
    ax.grid(True, which="major", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return marks


def make_figure(panels, out_path, label_extremes=False):
    """panels: list of (title, {metric: series}) in column order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    metrics = []
    for _, by_metric in panels:
        for m in by_metric:
            if m not in metrics:
                metrics.append(m)
    n_rows, n_cols = len(metrics), len(panels)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols + 1.2, 2.9 * n_rows + 0.6), squeeze=False, sharex="col",
        layout="constrained",
    )
    all_marks = set()
    for col, (title, by_metric) in enumerate(panels):
        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            series = by_metric.get(metric)
            if not series:
                ax.set_axis_off()
                continue
            all_marks.update(draw_panel(ax, series, label_extremes))
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (%)", fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("Total compute (FLOPs)", fontsize=9)
            ax.tick_params(labelsize=8)

    if all_marks:
        cmap = plt.get_cmap("Blues")
        sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "sft", [cmap(0.35), cmap(1.0)]), norm=LogNorm(min(all_marks), max(all_marks)))
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, shrink=0.8)
        cbar.set_label("SFT compute at fork (GPU-min)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    handles = [
        Line2D([], [], color="#8a8a8a", ls="--", marker="o", mfc="white", label="SFT only"),
        Line2D([], [], color=plt.get_cmap("Blues")(0.75), marker="o", mfc="white", label="GRPO from SFT mark (ring = pre-RL)"),
        Line2D([], [], color="black", label="Pareto frontier"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- CLI ---------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="SFT/RL compute frontier figure from W&B")
    parser.add_argument("--lineages", nargs="+", default=["l3b", "g4b", "l8b", "g12b"], choices=sorted(LINEAGES))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", default="budget_frontier", help="basename for the figure and CSV")
    parser.add_argument("--token-lengths", type=Path, default=TOKEN_LENGTHS_PATH)
    parser.add_argument("--refresh", action="store_true", help="re-fetch W&B history instead of using the cache")
    parser.add_argument("--label-extremes", action="store_true", help="also label the smallest and largest SFT marks")
    args = parser.parse_args(argv)

    token_lengths = json.loads(args.token_lengths.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.name}.csv"
    with csv_path.open("w", newline="") as fh:
        csv.writer(fh).writerow([
            "lineage", "metric", "run_id", "mark_minutes", "total_minutes",
            "sft_flops", "rl_flops", "total_flops", "rl_fraction", "value",
        ])

    panels = []
    for name in args.lineages:
        lineage = LINEAGES[name]
        raw = load_or_fetch(name, lineage, args.out_dir, args.refresh)
        n_params = NON_EMBEDDING_PARAMS[lineage.model]
        mean_tokens = token_lengths[lineage.token_spec]["mean_tokens"]
        by_metric = {}
        for metric in lineage.metrics:
            series = build_series(raw, metric, n_params, mean_tokens)
            if series:
                by_metric[metric] = series
                write_points_csv(csv_path, name, metric, series)
        print(f"{name}: {len(raw)} runs, {sum(len(s.points) for s in by_metric.get(lineage.metrics[0], []))} "
              f"{lineage.metrics[0]} points, N={n_params:.3e}, mean SFT tokens/example={mean_tokens:.1f}")
        panels.append((lineage.title, by_metric))

    fig_path = args.out_dir / f"{args.name}.png"
    make_figure(panels, fig_path, args.label_extremes)
    print(f"wrote {fig_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
