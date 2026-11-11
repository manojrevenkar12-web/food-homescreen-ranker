"""Ranking features, two ways that must agree:

- `offline(...)` runs features.sql over the logs (DuckDB here, Spark SQL in the nightly job) for training;
- `online(...)` computes the same values from the platform's live counters when a page is served.

tests/test_features.py checks that both give the same numbers for the same sessions, so the ranker is
trained on exactly what it will see at serving time.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from fhr.config import ROOT
from fhr.world.city import Catalog
from fhr.world.sim import Context, Logs, Observed

FEATURES = [
    "cm_orders",
    "cm_days_since",
    "c_orders",
    "c_cuisine_share",
    "c_price_gap",
    "m_orders_7d",
    "m_orders",
    "m_late_rate",
    "m_rating",
    "m_ctr",
    "m_age_days",
    "price",
    "promoted",
    "commission",
    "cuisine",
    "morning_fit",
    "distance",
    "eta",
    "load",
    "hour",
]
BREAKFAST = ("breakfast", "dessert", "grocery")  # the cuisines the menu team tags as breakfast


@lru_cache
def sql() -> str:
    return (ROOT / "src" / "fhr" / "features" / "features.sql").read_text(encoding="utf-8")


def merchants_table(catalog: Catalog) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "merchant": np.arange(catalog.n),
            "cuisine": catalog.cuisine,
            "price": catalog.price,
            "promoted": catalog.promoted.astype(int),
            "commission": catalog.commission,
            "joined_day": catalog.joined_day,
            "breakfast": np.isin(np.array(catalog.cuisines)[catalog.cuisine], BREAKFAST).astype(int),
        }
    )


def tables(logs: Logs, catalog: Catalog, candidates: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    cand = candidates if candidates is not None else logs.impressions[["sid", "merchant", "distance", "eta", "load"]]
    return {
        "candidates": cand,
        "sessions": logs.sessions[["sid", "day", "hour", "customer"]],
        "orders": logs.orders[["sid", "customer", "merchant", "late", "stars"]],
        "impressions": logs.impressions[["sid", "merchant", "clicked"]],
        "merchants": merchants_table(catalog),
    }


def offline(logs: Logs, catalog: Catalog, candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect()
    for name, df in tables(logs, catalog, candidates).items():
        con.register(name, df)
    out = con.execute(sql()).df()
    con.close()
    return out


def offline_spark(logs: Logs, catalog: Catalog, spark) -> pd.DataFrame:  # pragma: no cover - CI job "spark"
    for name, df in tables(logs, catalog).items():
        # pandas marks a missing rating as NaN; Spark would keep NaN as a number, DuckDB reads it as NULL
        clean = df.astype(object).where(df.notna(), None)
        spark.createDataFrame(clean, schema=_schema(df)).createOrReplaceTempView(name)
    return spark.sql(sql()).toPandas()


def _schema(df: pd.DataFrame):  # pragma: no cover - CI job "spark"
    from pyspark.sql import types as T

    def kind(dtype):
        if pd.api.types.is_bool_dtype(dtype):
            return T.BooleanType()
        if pd.api.types.is_integer_dtype(dtype):
            return T.LongType()
        if pd.api.types.is_float_dtype(dtype):
            return T.DoubleType()
        return T.StringType()

    return T.StructType([T.StructField(c, kind(df[c].dtype), True) for c in df.columns])


def online(ctx: Context, obs: Observed, catalog: Catalog, merchants: np.ndarray) -> np.ndarray:
    """Features for `merchants` (a subset of ctx.available) at serving time, in FEATURES order."""
    pos = {int(m): i for i, m in enumerate(ctx.available)}
    idx = np.array([pos[int(m)] for m in merchants], dtype=int)
    c, day = ctx.customer, ctx.day
    cm = obs.orders_cm[c, merchants].astype(float)
    last = obs.last_day_cm[c, merchants]
    past = obs.orders_cm[c].astype(float)
    c_orders = past.sum()
    if c_orders > 0:
        by_cuisine = np.bincount(catalog.cuisine, weights=past, minlength=len(catalog.cuisines))
        c_share = by_cuisine[catalog.cuisine[merchants]] / c_orders
        c_gap = catalog.price[merchants] - (past @ catalog.price) / c_orders
    else:
        c_share = np.zeros(len(merchants))
        c_gap = np.zeros(len(merchants))
    lo = max(0, day - 7)
    o7 = obs.daily_orders[lo:day, merchants].sum(axis=0)
    o_all = obs.daily_orders[:day, merchants].sum(axis=0)
    late = obs.daily_late[:day, merchants].sum(axis=0)
    stars_sum = obs.daily_stars_sum[:day, merchants].sum(axis=0)
    stars_n = obs.daily_stars_n[:day, merchants].sum(axis=0)
    impr = obs.daily_impr[:day, merchants].sum(axis=0)
    clicks = obs.daily_clicks[:day, merchants].sum(axis=0)
    breakfast = np.isin(np.array(catalog.cuisines)[catalog.cuisine[merchants]], BREAKFAST)
    cols = [
        cm,
        np.where(last < 0, 99, day - last).astype(float),
        np.full(len(merchants), c_orders),
        c_share,
        c_gap,
        o7,
        o_all,
        np.where(o_all > 0, late / np.maximum(o_all, 1), 0.0),
        (stars_sum + 3.6 * 5) / (stars_n + 5),
        (clicks + 1.0) / (impr + 10.0),
        (day - catalog.joined_day[merchants]).astype(float),
        catalog.price[merchants].astype(float),
        catalog.promoted[merchants].astype(float),
        catalog.commission[merchants],
        catalog.cuisine[merchants].astype(float),
        ((ctx.hour < 11) & breakfast).astype(float),
        ctx.distance[idx],
        ctx.eta[idx],
        ctx.load[idx],
        np.full(len(merchants), float(ctx.hour)),
    ]
    return np.column_stack(cols)
