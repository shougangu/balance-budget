# ABOUTME: Verifies a saved SFT dataset loads with non-empty train/test splits and the expected row shape.
# ABOUTME: Exits non-zero on any problem so a job gate can depend on it.

import sys

from datasets import load_from_disk

from tuning.config import DATASETS_DIR


def main(name: str) -> int:
    ds = load_from_disk(f"{DATASETS_DIR}/{name}")
    train, test = ds["train"], ds["test"]
    if train.num_rows == 0 or test.num_rows == 0:
        print(f"{name}: empty split train={train.num_rows} test={test.num_rows}")
        return 1
    roles = [m["role"] for m in train[0]["messages"]]
    if roles != ["system", "user", "assistant"]:
        print(f"{name}: unexpected roles {roles}")
        return 1
    print(f"{name}: ok train={train.num_rows} test={test.num_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
