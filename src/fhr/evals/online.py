"""Run policies on the same simulated future and score them against the truth.

Scoring reads the truth (which merchants are new, what really happened); policies never do.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fhr.config import platform
from fhr.policies.heuristic import Heuristic
from fhr.world.city import City
from fhr.world.sim import Logs, World, schedule


def history(city: City, seed: int) -> tuple[World, Logs]:
    """The logging period: today's home screen, with a share of exploration sessions."""
    days = city.config["time"]["history_days"]
    world = World(city)
    h = Heuristic(city.catalog, seed)
    rate = platform()["policy"]["explore_rate"]
    logs = world.run(
        schedule(city, 0, days, seed),
        {"heuristic": h, "heuristic:explore": h.explore},
        lambda c: "heuristic",
        seed,
        explore=lambda sid: np.random.default_rng([seed, sid, 9]).random() < rate,
    )
    return world, logs


def eval_sessions(city: City, seed: int) -> pd.DataFrame:
    t = city.config["time"]
    return schedule(city, t["history_days"], t["history_days"] + t["eval_days"], seed + 1)


def run_policy(world: World, sessions: pd.DataFrame, policy: Callable, seed: int, name: str = "policy") -> Logs:
    return world.fork().run(sessions, {name: policy}, lambda c: name, seed + 2)


def metrics(logs: Logs, city: City) -> dict[str, float]:
    s, o = logs.sessions, logs.orders
    n = len(s)
    cat = city.catalog
    new_days = platform()["policy"]["new_days"]
    o = o.merge(s[["sid", "day"]], on="sid")
    is_new = (o.day - cat.joined_day[o.merchant.to_numpy()]) < new_days
    counts = o.merchant.value_counts()
    top = int(np.ceil(0.1 * cat.n))
    return {
        "sessions": n,
        "orders_per_session": len(o) / n,
        "gmv_per_session": o.basket.sum() / n,
        "commission_per_session": o.commission.sum() / n,
        "late_share": float(o.late.mean()) if len(o) else 0.0,
        "new_merchant_share": float(is_new.mean()) if len(o) else 0.0,
        "top10_share": float(counts.iloc[:top].sum() / max(1, len(o))),
        "search_share": float((o.via == "search").mean()) if len(o) else 0.0,
        "merchants_with_orders": int(o.merchant.nunique()),
    }


def per_customer(logs: Logs) -> pd.DataFrame:
    s = logs.sessions.groupby("customer").size().rename("sessions")
    o = logs.orders.groupby("customer").size().rename("orders")
    g = logs.orders.groupby("customer").basket.sum().rename("gmv")
    return pd.concat([s, o, g], axis=1).fillna(0.0)


def paired_lift(a: Logs, b: Logs, reps: int = 1000, seed: int = 0) -> dict[str, float]:
    """b versus a on orders per session, customers resampled together (same sessions in both)."""
    pa, pb = per_customer(a), per_customer(b).reindex(per_customer(a).index, fill_value=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pa), size=(reps, len(pa)))
    sa, oa, ob = pa.sessions.to_numpy(), pa.orders.to_numpy(), pb.orders.to_numpy()
    base = oa.sum() / sa.sum()
    diff = (ob.sum() - oa.sum()) / sa.sum()
    boot = (ob[idx].sum(1) - oa[idx].sum(1)) / sa[idx].sum(1)
    return {
        "diff": float(diff),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "relative": float(diff / base),
    }
