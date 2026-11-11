"""The held-out city, once. Its config and the decision code were pinned in config/heldout.lock first.

Usage: python scripts/heldout.py
"""

from __future__ import annotations

import hashlib
import json
import sys

from fhr.config import platform
from fhr.evals.experiments import guardrails
from fhr.evals.online import eval_sessions, metrics, paired_lift
from fhr.policies.heuristic import Heuristic
from fhr.policies.learned import PageAssembler, Reranked

from _common import REPORTS, ROOT, pipeline, write

LOCK = ROOT / "config" / "heldout.lock"


def main() -> int:
    lock = json.loads(LOCK.read_text())
    changed = [p for p, h in lock["files"].items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    if changed:
        print(f"refusing: pinned files changed: {changed}", file=sys.stderr)
        return 1
    if (REPORTS / "heldout.json").exists():
        print("refusing: the held-out run already happened", file=sys.stderr)
        return 1
    seed = platform()["eval"]["heldout_seed"]
    city, world, logs, bundle = pipeline(seed, "world_heldout")
    sessions = eval_sessions(city, seed)
    runs = {}
    for name, p in [
        ("current", Heuristic(city.catalog, seed)),
        ("reranked", Reranked(city.catalog, bundle)),
        ("page", PageAssembler(city.catalog, bundle)),
    ]:
        runs[name] = world.fork().run(sessions, {name: p}, lambda c, n=name: n, seed + 2)
        print(name, metrics(runs[name], city), flush=True)
    new_days = platform()["policy"]["new_days"]
    write(
        "heldout.json",
        {
            "lock": lock,
            "training": bundle.report,
            "metrics": {k: metrics(v, city) for k, v in runs.items()},
            "lift_vs_current": {k: paired_lift(runs["current"], v) for k, v in runs.items() if k != "current"},
            "guardrails": {k: guardrails(v, city.catalog, new_days) for k, v in runs.items()},
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
