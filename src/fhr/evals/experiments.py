"""A/B test designs for a home-screen change, scored against the true effect of rolling it out to everyone.

Kitchens are shared. In a customer-split test the treatment sends orders to kitchens that control customers
also use, so each arm changes the other's delivery times. A switchback gives the whole city one policy per
time block, so the kitchens see what a full rollout would do; the first hour of each block is dropped,
because orders from the previous block still load the kitchens for up to an hour.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from fhr.world.sim import Logs, World


def _hash(salt: str, key: int) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def customer_split(
    world: World, sessions: pd.DataFrame, control: Callable, treatment: Callable, salt: str, seed: int
) -> Logs:
    w = world.fork()
    return w.run(
        sessions,
        {"control": control, "treatment": treatment},
        lambda c: "treatment" if _hash(salt, c) < 0.5 else "control",
        seed,
    )


def switchback(
    world: World, sessions: pd.DataFrame, control: Callable, treatment: Callable, salt: str, seed: int, block_hours: int
) -> Logs:
    w = world.fork()
    block = (sessions.minute // (60 * block_hours)).to_numpy()
    arm = {int(b): ("treatment" if _hash(salt, int(b)) < 0.5 else "control") for b in np.unique(block)}
    by_sid = dict(zip(sessions.sid.to_numpy(), block, strict=True))
    current = {"sid": None}

    def assign(_customer: int) -> str:
        return arm[by_sid[current["sid"]]]

    policies = {"control": control, "treatment": treatment}
    for s in sessions.itertuples(index=False):
        current["sid"] = int(s.sid)
        ctx = w.context(int(s.sid), int(s.day), int(s.minute), int(s.customer))
        name = assign(int(s.customer))
        w.step(ctx, policies[name](ctx, w.obs), False, name, seed)
    return w.logs()


def analyse_customer_split(logs: Logs, pre: pd.DataFrame, reps: int = 1000, seed: int = 0, cuped: bool = True) -> dict:
    """Difference in orders per session, treatment minus control, customers resampled.

    Orders per session is a ratio, so CUPED works on its linearisation: each customer's orders minus the
    pooled rate times their sessions, adjusted by what their pre-period order rate predicts for the same
    number of sessions. `pre` has each customer's pre-period `orders` and `sessions`."""
    s = logs.sessions
    per = s.groupby(["customer", "policy"]).agg(sessions=("sid", "size"), orders=("ordered", "sum")).reset_index()
    if cuped and len(pre):
        base = pre.orders.sum() / max(pre.sessions.sum(), 1)
        rate = (pre.orders / pre.sessions.clip(lower=1)).reindex(per.customer).fillna(base).to_numpy()
        lin = per.orders - per.orders.sum() / per.sessions.sum() * per.sessions
        cov = (rate - base) * per.sessions.to_numpy()
        theta = np.cov(lin, cov)[0, 1] / max(np.var(cov, ddof=1), 1e-12)
        per["y"] = per.orders - theta * cov
    else:
        per["y"] = per.orders.astype(float)
    t, c = per[per.policy == "treatment"], per[per.policy == "control"]

    def est(tt, cc):
        return tt.y.sum() / tt.sessions.sum() - cc.y.sum() / cc.sessions.sum()

    rng = np.random.default_rng(seed)
    boot = [est(t.iloc[rng.integers(0, len(t), len(t))], c.iloc[rng.integers(0, len(c), len(c))]) for _ in range(reps)]
    return {
        "diff": float(est(t, c)),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "se": float(np.std(boot)),
    }


def analyse_switchback(
    logs: Logs,
    block_hours: int,
    burn_in_minutes: int = 60,
    reps: int = 1000,
    seed: int = 0,
    orders: pd.DataFrame | None = None,
) -> dict:
    s = logs.sessions.copy()
    s["block"] = s.minute // (60 * block_hours)
    s = s[(s.minute % (60 * block_hours)) >= burn_in_minutes]
    b = s.groupby(["block", "policy", "day"]).agg(sessions=("sid", "size"), orders=("ordered", "sum")).reset_index()
    b["rate"] = b.orders / b.sessions

    def est(df):
        # day fixed effects: compare arms within the same day, weight by sessions
        out, w = 0.0, 0.0
        for _, g in df.groupby("day"):
            t, c = g[g.policy == "treatment"], g[g.policy == "control"]
            if len(t) == 0 or len(c) == 0:
                continue
            d = t.orders.sum() / t.sessions.sum() - c.orders.sum() / c.sessions.sum()
            n = g.sessions.sum()
            out += d * n
            w += n
        return out / w if w else float("nan")

    rng = np.random.default_rng(seed)
    days = b.day.unique()
    boot = []
    for _ in range(reps):
        parts = []
        for d in days:
            g = b[b.day == d]
            parts.append(g.iloc[rng.integers(0, len(g), len(g))])
        boot.append(est(pd.concat(parts)))
    boot = np.array([x for x in boot if np.isfinite(x)])
    return {
        "diff": float(est(b)),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "se": float(np.std(boot)),
        "blocks": int(b.block.nunique()),
    }


def guardrails(logs: Logs, catalog, new_days: int) -> dict[str, dict[str, float]]:
    """Per arm: late share, share of orders to new merchants, share of orders on the top 10% of merchants."""
    o = logs.orders.merge(logs.sessions[["sid", "policy", "day"]], on="sid")
    out = {}
    top = int(np.ceil(0.1 * catalog.n))
    for arm, g in o.groupby("policy"):
        counts = g.merchant.value_counts()
        out[arm] = {
            "late_share": float(g.late.mean()),
            "new_merchant_share": float(((g.day - catalog.joined_day[g.merchant.to_numpy()]) < new_days).mean()),
            "top10_share": float(counts.iloc[:top].sum() / len(g)),
        }
    return out
