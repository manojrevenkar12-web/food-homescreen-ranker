"""Train on a city's logging period and save the bundle (used by the container smoke test in CI).

Usage: python scripts/train_bundle.py OUT_DIR [--world world_small --seed 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fhr.serve.bundle import save

from _common import pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--world", default="world_small")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    _, _, _, bundle = pipeline(a.seed, a.world)
    print(save(bundle, a.out))


if __name__ == "__main__":
    main()
