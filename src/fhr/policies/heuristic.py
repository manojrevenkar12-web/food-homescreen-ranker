"""The home screen as it runs today: a fixed row template, each row sorted by a hand-weighted personal score.

With a small probability a session is served an exploration page: the same rows in random order with their
cards shuffled. Those sessions carry known propensities and make unbiased offline evaluation possible.
"""

from __future__ import annotations

import numpy as np

from fhr.config import platform
from fhr.world.city import Catalog
from fhr.world.sim import MAX_CARDS, Context, Observed, Row


def popularity(obs: Observed, day: int, window: int = 7) -> np.ndarray:
    lo = max(0, day - window)
    return obs.daily_orders[lo:day].sum(axis=0).astype(float)


def favourite_cuisine(obs: Observed, catalog: Catalog, customer: int) -> int | None:
    past = obs.orders_cm[customer]
    if past.sum() == 0:
        return None
    counts = np.bincount(catalog.cuisine, weights=past, minlength=len(catalog.cuisines))
    return int(counts.argmax())


def affinity_score(ctx: Context, obs: Observed, catalog: Catalog) -> np.ndarray:
    """Today's personalised sort: popularity, the customer's cuisine mix, past orders, distance and ETA,
    with weights a team would set by hand."""
    a = ctx.available
    pop = popularity(obs, ctx.day)[a]
    past = obs.orders_cm[ctx.customer].astype(float)
    share = np.zeros(len(a))
    if past.sum() > 0:
        by_cuisine = np.bincount(catalog.cuisine, weights=past, minlength=len(catalog.cuisines))
        share = by_cuisine[catalog.cuisine[a]] / past.sum()
    return (
        np.log1p(pop)
        + 1.5 * share
        + 0.8 * np.log1p(obs.orders_cm[ctx.customer, a])
        - 0.15 * ctx.distance
        - 0.03 * ctx.eta
    )


def heuristic_rows(ctx: Context, obs: Observed, catalog: Catalog) -> list[Row]:
    a = ctx.available
    pop = affinity_score(ctx, obs, catalog)
    new_days = platform()["policy"]["new_days"]
    rows: list[Row] = []
    past = obs.orders_cm[ctx.customer, a]
    if past.sum() > 0:
        idx = np.argsort(-past, kind="stable")
        rows.append(Row("order_again", [int(a[i]) for i in idx if past[i] > 0][:MAX_CARDS]))
    rows.append(Row("recommended", [int(a[i]) for i in np.argsort(-pop, kind="stable")[:MAX_CARDS]]))
    promo = np.flatnonzero(catalog.promoted[a])
    rows.append(Row("deals", [int(a[i]) for i in promo[np.argsort(-pop[promo], kind="stable")]][:MAX_CARDS]))
    rows.append(Row("fast", [int(a[i]) for i in np.argsort(ctx.eta, kind="stable")[:MAX_CARDS]]))
    new = np.flatnonzero(ctx.day - catalog.joined_day[a] < new_days)
    rows.append(
        Row("new", [int(a[i]) for i in new[np.argsort(-catalog.joined_day[a][new], kind="stable")]][:MAX_CARDS])
    )
    fav = favourite_cuisine(obs, catalog, ctx.customer)
    if fav is not None:
        same = np.flatnonzero(catalog.cuisine[a] == fav)
        rows.append(
            Row(
                f"cuisine:{catalog.cuisines[fav]}",
                [int(a[i]) for i in same[np.argsort(-pop[same], kind="stable")]][:MAX_CARDS],
            )
        )
    return [r for r in rows if r.merchants]


def shuffled(rows: list[Row], rng: np.random.Generator) -> list[Row]:
    order = rng.permutation(len(rows))
    out = []
    for i in order:
        ms = list(rows[i].merchants)
        rng.shuffle(ms)
        out.append(Row(rows[i].kind, ms))
    return out


class Heuristic:
    name = "heuristic"

    def __init__(self, catalog: Catalog, seed: int = 0):
        self.catalog = catalog
        self.seed = seed

    def __call__(self, ctx: Context, obs: Observed) -> list[Row]:
        return heuristic_rows(ctx, obs, self.catalog)

    def explore(self, ctx: Context, obs: Observed) -> list[Row]:
        return shuffled(heuristic_rows(ctx, obs, self.catalog), np.random.default_rng([self.seed, ctx.sid, 1]))
