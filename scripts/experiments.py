"""A/B test designs for the page assembler against today's screen, against the true global effect.

Usage: python scripts/experiments.py
"""

from __future__ import annotations

import json
import time

from fhr.config import platform
from fhr.evals.experiments import analyse_customer_split, analyse_switchback, customer_split, guardrails, switchback
from fhr.evals.online import eval_sessions, metrics, paired_lift
from fhr.policies.heuristic import Heuristic
from fhr.policies.learned import PageAssembler, Reranked

from _common import pipeline, write

BLOCK_HOURS = 3


def main() -> None:
    cfg = platform()["experiment"]
    seed = platform()["eval"]["seeds"][0]
    city, world, logs, bundle = pipeline(seed)
    sessions = eval_sessions(city, seed)
    current = Heuristic(city.catalog, seed)
    pre = logs.sessions.groupby("customer").ordered.sum()
    new_days = platform()["policy"]["new_days"]
    out = {"seed": seed, "block_hours": BLOCK_HOURS, "treatments": {}}
    for name, treatment in [
        ("page", PageAssembler(city.catalog, bundle)),
        ("reranked", Reranked(city.catalog, bundle)),
    ]:
        t = time.perf_counter()
        base = world.fork().run(sessions, {"current": current}, lambda c: "current", seed + 2)
        full = world.fork().run(sessions, {name: treatment}, lambda c, n=name: n, seed + 2)
        truth = paired_lift(base, full)
        res = {
            "truth": truth,
            "global": {"current": metrics(base, city), name: metrics(full, city)},
            "customer_split": [],
            "customer_split_no_cuped": [],
            "switchback": [],
        }
        for r in range(cfg["replications"]):
            cs = customer_split(world, sessions, current, treatment, f"cs{r}", seed + 2)
            res["customer_split"].append(
                analyse_customer_split(cs, pre, seed=r) | {"guardrails": guardrails(cs, city.catalog, new_days)}
            )
            res["customer_split_no_cuped"].append(analyse_customer_split(cs, pre, seed=r, cuped=False))
            sb = switchback(world, sessions, current, treatment, f"sb{r}", seed + 2, BLOCK_HOURS)
            res["switchback"].append(
                analyse_switchback(sb, BLOCK_HOURS, seed=r) | {"guardrails": guardrails(sb, city.catalog, new_days)}
            )
            print(
                f"  {name} rep {r}: split {res['customer_split'][-1]['diff']:+.4f} "
                f"switchback {res['switchback'][-1]['diff']:+.4f} truth {truth['diff']:+.4f}",
                flush=True,
            )
        out["treatments"][name] = res
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(res, default=float))
        print(f"{name}: {time.perf_counter() - t:.0f} s", flush=True)
    write("experiments.json", out)


if __name__ == "__main__":
    main()
