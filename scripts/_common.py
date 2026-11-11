"""Shared helpers for the scripts: the pipeline per seed and report files."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REPORTS = ROOT / "reports"


def write(name: str, data: Any) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / name).write_text(json.dumps(data, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote reports/{name}")


def read(name: str) -> Any:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def pipeline(seed: int, world_config: str = "world"):
    """City, the logging period, and the models trained on it."""
    from fhr.evals.online import history
    from fhr.train import train
    from fhr.world.city import build

    t = time.perf_counter()
    city = build(seed, world_config)
    world, logs = history(city, seed)
    c = city.config
    bundle = train(
        logs,
        city.catalog,
        city.customers,
        c["city"]["grid"] ** 2,
        c["time"]["history_days"],
        c["city"]["delivery_radius_km"],
    )
    print(f"seed {seed}: history and training in {time.perf_counter() - t:.0f} s", flush=True)
    return city, world, logs, bundle
