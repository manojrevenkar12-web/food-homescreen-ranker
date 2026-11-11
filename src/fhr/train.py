"""The training pipeline the Airflow DAG runs every night (dags/home_ranker_daily.py):

  logs -> features (features.sql) -> two-tower -> ranker (+ calibration) -> sequential challenger
       -> attention estimates from exploration sessions -> offline report -> bundle

Temporal splits keep it honest: the two-tower that produces the ranker's `tt_score` feature is trained on days
before the ranker's training days, so the feature carries no knowledge of the orders it is asked to rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fhr.config import platform
from fhr.features.build import offline
from fhr.models.ranker import Ranker, ndcg_at
from fhr.models.sequential import Sequential, count_scores
from fhr.models.two_tower import TwoTower, examples_from_logs
from fhr.world.city import Catalog, Customers
from fhr.world.sim import MAX_CARDS, MAX_ROWS, Logs


@dataclass
class Bundle:
    two_tower: TwoTower
    ranker: Ranker
    sequential: Sequential | None
    row_decay: float
    card_decay: float
    report: dict[str, Any] = field(default_factory=dict)


def histories_at(logs: Logs, sids: np.ndarray) -> dict[int, list[int]]:
    """Each session's customer's order history before that session."""
    o = logs.orders.sort_values("sid")[["sid", "customer", "merchant"]].to_numpy()
    s = logs.sessions.set_index("sid").loc[np.sort(np.unique(sids)), "customer"]
    out, hist, k = {}, {}, 0
    for sid, cust in s.items():
        while k < len(o) and o[k, 0] < sid:
            hist.setdefault(int(o[k, 1]), []).append(int(o[k, 2]))
            k += 1
        out[int(sid)] = list(hist.get(int(cust), []))
    return out


def attention(logs: Logs) -> tuple[float, float]:
    """Row and card attention from exploration sessions, where row order and card order were random."""
    ex = logs.impressions[logs.impressions.explore]
    if ex.empty:
        cfg = platform()["page"]
        return cfg["row_decay"], 0.72
    rows = ex.groupby("sid").row.max()
    reach = np.array([(rows >= r).mean() for r in range(MAX_ROWS)])
    row_decay = float(np.exp(np.mean(np.diff(np.log(np.clip(reach[:5], 1e-6, None))))))
    per_row = ex.groupby(["sid", "row"]).pos.max()
    vis = 2
    cards = np.array([(per_row >= j).mean() for j in range(vis - 1, MAX_CARDS)])
    card_decay = float(np.exp(np.mean(np.diff(np.log(np.clip(cards, 1e-6, None))))))
    return row_decay, card_decay


def add_tt(df: pd.DataFrame, tt: TwoTower, hist: dict[int, list[int]], zone: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    scores = np.empty(len(df))
    for sid, idx in df.groupby("sid").indices.items():
        g = df.iloc[idx]
        scores[idx] = tt.scores(
            hist[int(sid)], int(zone[g.customer.iloc[0]]), int(g.hour.iloc[0]), g.merchant.to_numpy()
        )
    df["tt_score"] = scores
    return df


def train(logs: Logs, catalog: Catalog, customers: Customers, n_zones: int, days: int, radius: float) -> Bundle:
    customer_zone = customers.zone
    cfg = platform()
    valid_from = days - cfg["ranker"]["valid_days"]
    tt_from = days // 2
    examples = examples_from_logs(logs, customer_zone)
    day_of = logs.sessions.set_index("sid").day
    examples["day"] = day_of.loc[examples.sid].to_numpy()

    # ---- two-tower for the ranker's feature: trained on the first half only
    tt_early = TwoTower(catalog, n_zones).fit(examples[examples.day < tt_from])

    # ---- ranker data: viewed impressions in the second half, features point in time
    feats = offline(logs, catalog)
    imp = logs.impressions[["sid", "merchant", "clicked", "explore", "row", "pos"]]
    feats = feats.merge(imp, on=["sid", "merchant"])
    page_orders = logs.orders[logs.orders.via == "page"][["sid", "merchant"]].assign(ordered=True)
    feats = feats.merge(page_orders, on=["sid", "merchant"], how="left").fillna({"ordered": False})
    feats = feats[feats.day >= tt_from]
    hist = histories_at(logs, feats.sid.to_numpy())
    feats = add_tt(feats, tt_early, hist, customer_zone)
    train_rows = feats[(feats.day < valid_from) & ~feats.explore]
    valid = feats[feats.day >= valid_from]
    ranker = Ranker().fit(train_rows, valid[valid.explore] if valid.explore.any() else valid)

    # ---- offline report on the held-out days
    valid = valid.assign(model=ranker.score(valid), served=-(valid.row * 100 + valid.pos), popularity=valid.m_orders_7d)
    report: dict[str, Any] = {
        "ndcg8": {k: ndcg_at(valid, k) for k in ("model", "served", "popularity", "tt_score")},
        "ndcg8_explore": {k: ndcg_at(valid[valid.explore], k) for k in ("model", "served", "popularity")},
        "rows": {"train": int(len(train_rows)), "valid": int(len(valid))},
    }

    # ---- retrieval quality: next orders in the valid days, among merchants that deliver there
    tt_valid = TwoTower(catalog, n_zones).fit(examples[examples.day < valid_from])
    report["retrieval"] = retrieval_report(tt_valid, examples, catalog, customers, radius, valid_from)

    # ---- sequential challenger for the order-again row
    seq_hist = examples[examples.history.map(len) > 0]
    seq = Sequential(catalog.n).fit(
        list(seq_hist[seq_hist.day < valid_from].history), list(seq_hist[seq_hist.day < valid_from].merchant)
    )
    vs = seq_hist[seq_hist.day >= valid_from]
    r_seq = np.mean([m in np.argsort(-seq.scores(list(h)))[:10] for h, m in zip(vs.history, vs.merchant, strict=True)])
    r_cnt = np.mean(
        [m in np.argsort(-count_scores(list(h), catalog.n))[:10] for h, m in zip(vs.history, vs.merchant, strict=True)]
    )
    use_seq = bool(r_seq - r_cnt >= cfg["sequential"]["min_gain_recall10"])
    report["sequential"] = {
        "recall10_gru": float(r_seq),
        "recall10_counts": float(r_cnt),
        "ships": use_seq,
        "orders": int(len(vs)),
    }

    # ---- final models for serving: refit on every day
    tt_final = TwoTower(catalog, n_zones).fit(examples)
    if use_seq:
        seq = Sequential(catalog.n).fit(list(seq_hist.history), list(seq_hist.merchant))
    row_decay, card_decay = attention(logs)
    report["attention"] = {"row_decay": row_decay, "card_decay": card_decay}
    return Bundle(tt_final, ranker, seq if use_seq else None, row_decay, card_decay, report)


def retrieval_report(
    tt: TwoTower,
    examples: pd.DataFrame,
    catalog: Catalog,
    customers: Customers,
    radius: float,
    valid_from: int,
    ks: tuple[int, ...] = (20, 50, 120),
) -> dict[str, Any]:
    v = examples[examples.day >= valid_from]
    pop = examples[examples.day < valid_from].merchant.value_counts().reindex(range(catalog.n), fill_value=0).to_numpy()
    seen_before = set(examples[examples.day < valid_from].merchant)
    out = {f"tt@{k}": [] for k in ks} | {f"pop@{k}": [] for k in ks}
    cold = {f"tt@{k}": [] for k in ks} | {f"pop@{k}": [] for k in ks}
    for r in v.itertuples(index=False):
        d = np.hypot(catalog.x - customers.x[r.customer], catalog.y - customers.y[r.customer])
        avail = np.flatnonzero((catalog.joined_day <= r.day) & (d <= radius))
        s = tt.scores(list(r.history), int(r.zone), int(r.hour), avail)
        rank_tt = avail[np.argsort(-s)]
        rank_pop = avail[np.argsort(-pop[avail], kind="stable")]
        target = out if r.merchant in seen_before else cold
        for k in ks:
            target[f"tt@{k}"].append(r.merchant in rank_tt[:k])
            target[f"pop@{k}"].append(r.merchant in rank_pop[:k])
    return {
        "all": {k: float(np.mean(x)) for k, x in out.items() if x},
        "cold_start": {k: float(np.mean(x)) for k, x in cold.items() if x},
        "orders": int(len(v)),
        "cold_start_orders": int(sum(1 for m in v.merchant if m not in seen_before)),
    }
