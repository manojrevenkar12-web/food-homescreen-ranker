"""Serving latency through the HTTP service, and sample pages for the dashboard.

Usage: python scripts/serve_bench.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from fhr.config import platform
from fhr.evals.online import eval_sessions
from fhr.policies.heuristic import Heuristic
from fhr.serve.app import Replica, create_app
from fhr.serve.bundle import load, save

from _common import pipeline, write

PREFIX = {
    "pizza": "Pizzeria",
    "burger": "Burger Bar",
    "sushi": "Sushi",
    "indian": "Curry House",
    "vietnamese": "Pho",
    "doner": "Döner",
    "italian": "Trattoria",
    "chinese": "Wok",
    "thai": "Thai Kitchen",
    "mexican": "Taqueria",
    "salad": "Bowl & Co",
    "breakfast": "Morning Café",
    "dessert": "Gelato",
    "korean": "Seoul Grill",
    "georgian": "Khachapuri",
    "grocery": "Mini Market",
}


def name(catalog, m: int) -> str:
    return f"{PREFIX[catalog.cuisines[catalog.cuisine[m]]]} {m}"


def main() -> None:
    seed = platform()["eval"]["seeds"][0]
    city, world, _, bundle = pipeline(seed)
    with tempfile.TemporaryDirectory() as d:
        digest = save(bundle, Path(d))
        bundle = load(Path(d), city.catalog, city.config["city"]["grid"] ** 2)  # serve what was saved
    client = TestClient(create_app(Replica(city, world.fork(), bundle)))
    sessions = eval_sessions(city, seed).head(600)
    ms = []
    for s in sessions.itertuples():
        t = time.perf_counter()
        r = client.post("/invocations", json={"customer": int(s.customer), "minute": int(s.minute)})
        ms.append((time.perf_counter() - t) * 1000)
        assert r.status_code == 200
    write(
        "serving.json",
        {
            "requests": len(ms),
            "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)),
            "bundle_digest": digest,
            "note": "end to end through the HTTP service, one CPU core",
        },
    )

    # three customers at dinner: today's screen and the assembled one, for the dashboard
    current = Heuristic(city.catalog)
    cat = city.catalog
    picks = []
    rng = np.random.default_rng(1)
    hist_counts = world.obs.orders_cm.sum(axis=1)
    for want in (0, 2, 8):
        c = int(rng.choice(np.flatnonzero(hist_counts >= want) if want else np.flatnonzero(hist_counts == 0)))
        minute = city.config["time"]["history_days"] * 1440 + 19 * 60 + 5
        ctx = world.context(-1, minute // 1440, minute, c)
        page = client.post("/v1/home", json={"customer": c, "minute": minute}).json()["rows"]
        today = current(ctx, world.obs)

        def card(m: int, ctx=ctx) -> dict:
            i = int(np.flatnonzero(ctx.available == m)[0])
            return {
                "id": m,
                "name": name(cat, m),
                "cuisine": cat.cuisines[cat.cuisine[m]],
                "eta": round(float(ctx.eta[i])),
                "promoted": bool(cat.promoted[m]),
                "new": bool(ctx.day - cat.joined_day[m] < platform()["policy"]["new_days"]),
                "load": round(float(ctx.load[i]), 2),
            }

        picks.append(
            {
                "customer": c,
                "past_orders": int(hist_counts[c]),
                "today": [{"kind": r.kind, "cards": [card(m) for m in r.merchants]} for r in today],
                "assembled": [{"kind": r["kind"], "cards": [card(m) for m in r["merchants"]]} for r in page],
            }
        )
    write("sample_pages.json", {"seed": seed, "minute": "day 15, 19:05", "customers": picks})


if __name__ == "__main__":
    main()
