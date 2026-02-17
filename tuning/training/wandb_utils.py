from typing import Any


def get_early_pairs(config: Any) -> list[list[float]]:
    early_tuples = getattr(config, "early_tuples", None)
    if not early_tuples:
        return []
    return [[int(p), float(t)] for p, t in early_tuples]


def early_pair_tag(early_pairs: list[list[float]]) -> str:
    if not early_pairs:
        return "early_pair:none"
    return "early_pair:" + ",".join(f"{int(p)}@{float(t):g}" for p, t in early_pairs)
