"""Every policy on the same simulated week, three cities, scored against the truth.

Usage: python scripts/evaluate.py
"""

from __future__ import annotations

import json
import time

from fhr.config import platform
from fhr.evals.online import eval_sessions, metrics, paired_lift
from fhr.evals.oracle import Oracle
from fhr.policies.heuristic import Heuristic
from fhr.policies.learned import PageAssembler, Reranked

from _common import REPORTS, pipeline, write


def main() -> None:
    out = {"seeds": {}}
    for seed in platform()["eval"]["seeds"]:
        part = REPORTS / "partial" / f"evaluate-{seed}.json"
        if part.exists():  # resumable: a city already scored is not run again
            out["seeds"][seed] = json.loads(part.read_text())
            continue
        city, world, logs, bundle = pipeline(seed)
        sessions = eval_sessions(city, seed)
        policies = {
            "current": Heuristic(city.catalog, seed),
            "reranked": Reranked(city.catalog, bundle),
            "page": PageAssembler(city.catalog, bundle),
            "page_no_congestion": PageAssembler(city.catalog, bundle, congestion=False),
            "page_no_explore": PageAssembler(city.catalog, bundle, explore=False),
            "page_no_gmv": PageAssembler(city.catalog, bundle, gmv=False),
        }
        runs = {}
        for name, p in [*policies.items(), ("oracle", None)]:
            w = world.fork()
            policy = Oracle(w) if name == "oracle" else p
            t = time.perf_counter()
            runs[name] = w.run(sessions, {name: policy}, lambda c, n=name: n, seed + 2)
            print(f"  {name}: {time.perf_counter() - t:.0f} s", flush=True)
        out["seeds"][seed] = {
            "training": bundle.report,
            "history": {
                "sessions": len(logs.sessions),
                "orders": len(logs.orders),
                "explore_sessions": int(logs.sessions.explore.sum()),
            },
            "metrics": {k: metrics(v, city) for k, v in runs.items()},
            "lift_vs_current": {k: paired_lift(runs["current"], v) for k, v in runs.items() if k != "current"},
        }
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(out["seeds"][seed], default=float))
    write("evaluate.json", out)


if __name__ == "__main__":
    main()
