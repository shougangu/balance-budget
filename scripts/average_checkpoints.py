# ABOUTME: Averages the weights of several training checkpoints into one model, the
# ABOUTME: final-model construction step the OpenMathInstruct-2 recipe uses.

"""Build one model from the elementwise mean of several checkpoints.

Under a constant learning rate the optimizer never settles; it orbits the floor
of the loss basin and each checkpoint sits somewhere on that orbit, displaced by
gradient noise. Averaging the weights cancels the displacements. This is the
same mechanism as stochastic weight averaging, and NVIDIA credits it with more
than two MATH points in OpenMathInstruct-2 (Appendix A.4).

Shards are streamed one at a time and accumulated in float32, so peak memory is
a couple of shards rather than the whole model times the checkpoint count.

Usage:
    python scripts/average_checkpoints.py --out tuning/models/<name>_avg6 \
        --num-checkpoints 6 <ckpt_dir> [<ckpt_dir> ...]
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
# Everything a served checkpoint needs besides the weights themselves.
SIDECAR_PATTERNS = ("*.json", "*.txt", "*.model", "*.jinja", "*.py")


def pick_equally_spaced(items, k):
    """Choose k entries spread evenly across items, always keeping both ends."""
    if k <= 0:
        raise ValueError(f"num_checkpoints must be positive, got {k}")
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[-1]]
    last = len(items) - 1
    return [items[round(i * last / (k - 1))] for i in range(k)]


def _shard_layout(checkpoint: Path) -> dict:
    """Map shard filename -> list of tensor keys it holds."""
    index_path = checkpoint / INDEX_FILE
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        layout = {}
        for key, shard in weight_map.items():
            layout.setdefault(shard, []).append(key)
        return layout
    if (checkpoint / SINGLE_FILE).is_file():
        return {SINGLE_FILE: None}  # None means "every key in the file"
    raise ValueError(f"No safetensors weights found in {checkpoint}")


def _check_layouts_match(checkpoints, layouts):
    reference = {shard: (None if keys is None else sorted(keys))
                 for shard, keys in layouts[0].items()}
    for checkpoint, layout in zip(checkpoints[1:], layouts[1:]):
        other = {shard: (None if keys is None else sorted(keys))
                 for shard, keys in layout.items()}
        if other != reference:
            raise ValueError(
                f"Checkpoint {checkpoint} has a different shard/key layout than "
                f"{checkpoints[0]}; averaging requires checkpoints from one run."
            )


def _average_shard(checkpoints, shard_name):
    """Average one shard across checkpoints, accumulating floats in float32."""
    shards = [load_file(str(checkpoint / shard_name)) for checkpoint in checkpoints]

    reference_keys = set(shards[0])
    for checkpoint, shard in zip(checkpoints[1:], shards[1:]):
        if set(shard) != reference_keys:
            missing = reference_keys.symmetric_difference(shard)
            raise ValueError(
                f"Checkpoint {checkpoint} has a different key set in {shard_name}: "
                f"{sorted(missing)[:5]}"
            )

    averaged = {}
    for key in shards[0]:
        first = shards[0][key]
        if not first.is_floating_point():
            # Integer buffers (position ids, masks) are structural, not learned.
            averaged[key] = shards[-1][key].clone()
            continue
        accumulator = first.to(torch.float32)
        for shard in shards[1:]:
            accumulator = accumulator + shard[key].to(torch.float32)
        averaged[key] = (accumulator / len(shards)).to(first.dtype)
    return averaged


def average_checkpoints(checkpoints, out_dir):
    """Write the elementwise mean of `checkpoints` to `out_dir`; returns out_dir."""
    checkpoints = [Path(c) for c in checkpoints]
    if not checkpoints:
        raise ValueError("No checkpoints given to average")

    layouts = [_shard_layout(c) for c in checkpoints]
    _check_layouts_match(checkpoints, layouts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for shard_name in sorted(layouts[0]):
        print(f"[average] {shard_name} over {len(checkpoints)} checkpoints", flush=True)
        save_file(_average_shard(checkpoints, shard_name), str(out_dir / shard_name))

    for pattern in SIDECAR_PATTERNS:
        for path in sorted(checkpoints[0].glob(pattern)):
            shutil.copy2(path, out_dir / path.name)

    print(f"[average] wrote {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint dirs, in training order")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-checkpoints", type=int, default=None,
                        help="Average this many, spread evenly across the given list.")
    args = parser.parse_args()

    checkpoints = args.checkpoints
    if args.num_checkpoints is not None:
        checkpoints = pick_equally_spaced(checkpoints, args.num_checkpoints)
    print(f"[average] averaging {len(checkpoints)}: {checkpoints}")
    average_checkpoints(checkpoints, args.out)


if __name__ == "__main__":
    main()
