"""Two learned home screens.

`Reranked` keeps today's row template and sorts each row with the ranker: the change most teams ship first.

`PageAssembler` builds the whole screen. It scores every candidate once (two-tower retrieval, then the
ranker, calibrated to P(order | the card is seen)), turns that into a value with the business objectives, and
then fills the screen greedily: at each step it places the row whose not-yet-shown merchants add the most
expected value at that position, given how attention falls down the screen and along a row. Then it checks
the guardrails: promoted contracts in the first three rows, and cards for new merchants.

    value = P(order) x (1 + gmv_weight x commission / median commission)
            x congestion discount (live kitchen load)          # do not send customers to a kitchen that is full
          + exploration bonus x P(order) x sqrt(prior / (prior + impressions))   # new merchants get looked at
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fhr.config import platform
from fhr.features.build import online
from fhr.models.sequential import count_scores
from fhr.policies.heuristic import heuristic_rows
from fhr.train import Bundle
from fhr.world.city import Catalog
from fhr.world.sim import MAX_CARDS, MAX_ROWS, Context, Observed, Row


@dataclass
class Scored:
    merchants: np.ndarray
    p: np.ndarray  # calibrated P(order | seen)
    value: np.ndarray  # p with the business objectives applied
    score: np.ndarray  # raw ranker score


def candidates(
    ctx: Context, obs: Observed, catalog: Catalog, bundle: Bundle
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Retrieval: the two-tower's top merchants, plus everything a row template may need."""
    a = ctx.available
    hist = obs.last_orders.get(ctx.customer, [])
    k = platform()["two_tower"]["retrieve"]
    tt_all = bundle.two_tower.scores(hist, ctx.zone, ctx.hour, a)
    keep = set(a[np.argsort(-tt_all)[:k]].tolist())
    keep |= set(a[obs.orders_cm[ctx.customer, a] > 0].tolist())
    keep |= set(a[ctx.day - catalog.joined_day[a] < platform()["policy"]["new_days"]].tolist())
    keep |= set(a[catalog.promoted[a]].tolist())
    ms = np.array(sorted(keep), dtype=int)
    pos = {int(m): i for i, m in enumerate(a)}
    return ms, tt_all[[pos[int(m)] for m in ms]], hist


def score(
    ctx: Context,
    obs: Observed,
    catalog: Catalog,
    bundle: Bundle,
    *,
    gmv: bool = True,
    congestion: bool = True,
    explore: bool = True,
) -> Scored:
    cfg = platform()["page"]
    ms, tt, hist = candidates(ctx, obs, catalog, bundle)
    X = online(ctx, obs, catalog, ms)
    raw = bundle.ranker.score(np.column_stack([X, tt]))
    p = bundle.ranker.p_order(raw)
    value = p.copy()
    if gmv:
        comm = catalog.commission[ms] * catalog.basket[ms]
        value *= 1 + cfg["gmv_weight"] * comm / np.median(catalog.commission * catalog.basket)
    if congestion:
        pos = {int(m): i for i, m in enumerate(ctx.available)}
        load = ctx.load[[pos[int(m)] for m in ms]]
        value *= np.exp(-cfg["congestion_strength"] * np.clip(load - cfg["congestion_start"], 0, None))
    if explore:
        impr = obs.daily_impr[: ctx.day + 1, ms].sum(axis=0)
        value += cfg["explore_bonus"] * p * np.sqrt(cfg["explore_prior"] / (cfg["explore_prior"] + impr))
    return Scored(ms, p, value, raw)


class Reranked:
    name = "reranked"

    def __init__(self, catalog: Catalog, bundle: Bundle):
        self.catalog, self.bundle = catalog, bundle

    def __call__(self, ctx: Context, obs: Observed) -> list[Row]:
        s = score(ctx, obs, self.catalog, self.bundle, gmv=False, congestion=False, explore=False)
        rank = {int(m): float(v) for m, v in zip(s.merchants, s.score, strict=True)}
        rows = []
        for row in heuristic_rows(ctx, obs, self.catalog):
            if row.kind == "recommended":
                top = s.merchants[np.argsort(-s.score)][:MAX_CARDS]
                rows.append(Row("for_you", [int(m) for m in top]))
            elif row.kind == "new":
                rows.append(row)
            else:
                rows.append(Row(row.kind, sorted(row.merchants, key=lambda m: -rank.get(m, -1e9))))
        return rows


class PageAssembler:
    name = "page"

    def __init__(
        self,
        catalog: Catalog,
        bundle: Bundle,
        *,
        gmv: bool = True,
        congestion: bool = True,
        explore: bool = True,
        guardrails: bool = True,
    ):
        self.catalog, self.bundle = catalog, bundle
        self.flags = {"gmv": gmv, "congestion": congestion, "explore": explore}
        self.guardrails = guardrails
        self.cfg = platform()["page"]

    def library(self, ctx: Context, obs: Observed, s: Scored, hist: list[int]) -> list[Row]:
        cat = self.catalog
        by_value = s.merchants[np.argsort(-s.value)]
        val = {int(m): float(v) for m, v in zip(s.merchants, s.value, strict=True)}
        rows = [Row("for_you", [int(m) for m in by_value])]
        past = [int(m) for m in s.merchants if obs.orders_cm[ctx.customer, m] > 0]
        if past:
            if self.bundle.sequential is not None:
                seq = self.bundle.sequential.scores(hist)
                past.sort(key=lambda m: -seq[m])
            else:
                cs = count_scores(hist, cat.n)
                past.sort(key=lambda m: -val[m] * (1 + cs[m]))
            rows.append(Row("order_again", past))
        cuisines = cat.cuisine[s.merchants]
        mass = np.bincount(cuisines, weights=s.value, minlength=len(cat.cuisines))
        for c in np.argsort(-mass)[:3]:
            ms = [int(m) for m in by_value if cat.cuisine[m] == c]
            if len(ms) >= 3:
                rows.append(Row(f"cuisine:{cat.cuisines[c]}", ms))
        new = [int(m) for m in by_value if ctx.day - cat.joined_day[m] < platform()["policy"]["new_days"]]
        if new:
            rows.append(Row("new", new))
        pos = {int(m): i for i, m in enumerate(ctx.available)}
        fast = [int(m) for m in by_value if ctx.eta[pos[int(m)]] <= 25]
        if fast:
            rows.append(Row("fast", fast))
        deals = [int(m) for m in by_value if cat.promoted[m]]
        if deals:
            rows.append(Row("deals", deals))
        return rows

    def __call__(self, ctx: Context, obs: Observed) -> list[Row]:
        s = score(ctx, obs, self.catalog, self.bundle, **self.flags)
        hist = obs.last_orders.get(ctx.customer, [])
        lib = self.library(ctx, obs, s, hist)
        val = {int(m): float(v) for m, v in zip(s.merchants, s.value, strict=True)}
        rd, cd = self.bundle.row_decay, self.bundle.card_decay
        card_w = np.array([1.0 if j < 2 else cd ** (j - 1) for j in range(MAX_CARDS)])
        placed: set[int] = set()
        page: list[Row] = []
        while lib and len(page) < min(MAX_ROWS, self.cfg["rows"]):
            best, best_gain, best_cards = None, -1.0, []
            for i, row in enumerate(lib):
                cards = [m for m in row.merchants if m not in placed][:MAX_CARDS]
                gain = float(sum(card_w[j] * val[m] for j, m in enumerate(cards)))
                if gain > best_gain:
                    best, best_gain, best_cards = i, gain, cards
            row = lib.pop(best)
            if not best_cards:
                break
            page.append(Row(row.kind, best_cards))
            placed.update(best_cards)
        _ = rd  # row attention multiplies every row equally at a given depth, so greedy order already maximises it
        if self.guardrails:
            page = self.enforce(page, s, ctx)
        return page

    def enforce(self, page: list[Row], s: Scored, ctx: Context) -> list[Row]:
        """Promoted contracts and new-merchant cards, at the least cost to the page."""
        cat, cfg = self.catalog, self.cfg
        by_value = [int(m) for m in s.merchants[np.argsort(-s.value)]]
        shown = {m for r in page for m in r.merchants}
        promoted_top3 = sum(cat.promoted[m] for r in page[:3] for m in r.merchants)
        if promoted_top3 < cfg["promoted_min_top3"] and len(page) >= 2:
            top3 = {m for r in page[:3] for m in r.merchants}
            extra = [m for m in by_value if cat.promoted[m] and m not in top3]
            for m in extra[: cfg["promoted_min_top3"] - promoted_top3]:
                for r in page[3:]:  # move it up rather than show it twice
                    if m in r.merchants:
                        r.merchants.remove(m)
                r = page[1]  # third card of the second row: on screen, not first
                cards = r.merchants[:2] + [m] + r.merchants[2:]
                while len(cards) > MAX_CARDS:  # make room by dropping the last card that is not promoted
                    drop = max(i for i, x in enumerate(cards) if not cat.promoted[x])
                    cards.pop(drop)
                r.merchants = cards
            page = [r for r in page if r.merchants]
            shown = {m for r in page for m in r.merchants}
        new_days = platform()["policy"]["new_days"]
        new_near = [m for m in by_value if ctx.day - cat.joined_day[m] < new_days]
        new_shown = sum(1 for r in page for m in r.merchants if ctx.day - cat.joined_day[m] < new_days)
        if new_near and new_shown < cfg["new_min_cards"] and not any(r.kind == "new" for r in page):
            row = Row("new", [m for m in new_near if m not in shown][:MAX_CARDS])
            if row.merchants:
                if len(page) >= self.cfg["rows"]:
                    page = page[:-1]
                page.insert(min(3, len(page)), row)
        return page
