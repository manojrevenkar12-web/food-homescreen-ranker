"""The simulated city. THIS MODULE HOLDS THE TRUTH: tastes, quality and sensitivities nobody logs.

Policies receive a `Catalog` (what the platform knows about each merchant) and the logs. They never import
this module's `Truth`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from fhr.config import ROOT


@lru_cache
def world_config(name: str = "world") -> dict[str, Any]:
    with open(ROOT / "config" / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


@dataclass
class Catalog:
    """What the platform knows: the merchant list as the merchant app and the ops team enter it."""

    n: int
    x: np.ndarray
    y: np.ndarray
    zone: np.ndarray
    cuisine: np.ndarray  # index into cuisines
    cuisines: list[str]
    price: np.ndarray  # 1..3
    prep: np.ndarray  # minutes, as configured by the merchant
    capacity: np.ndarray  # orders per hour the merchant says it can take
    commission: np.ndarray
    promoted: np.ndarray  # bool: paid placement contract
    joined_day: np.ndarray  # first day on the platform
    basket: np.ndarray  # mean basket in EUR


@dataclass
class Customers:
    """What the platform knows about a customer: where they are. Their history comes from the logs."""

    n: int
    x: np.ndarray
    y: np.ndarray
    zone: np.ndarray
    rate: np.ndarray  # sessions per day (needed to schedule sessions; not given to policies)


@dataclass
class Truth:
    taste: np.ndarray  # customers x dim
    flavour: np.ndarray  # merchants x dim
    quality: np.ndarray  # merchants
    price_sens: np.ndarray
    eta_sens: np.ndarray
    novelty: np.ndarray
    loyalty: np.ndarray
    true_capacity: np.ndarray  # what the kitchen really handles (differs from the configured number)


@dataclass
class City:
    catalog: Catalog
    customers: Customers
    truth: Truth
    seed: int
    config: dict[str, Any]


def build(seed: int, config: str = "world") -> City:
    cfg = world_config(config)
    rng = np.random.default_rng(seed)
    city, m, c = cfg["city"], cfg["merchants"], cfg["customers"]
    side = city["grid"] * city["zone_km"]
    dim = city["dim"]
    cuisines = m["cuisines"]
    cw = np.array(m["cuisine_weights"], dtype=float)
    cw /= cw.sum()
    cuisine_emb = rng.normal(0, 1, size=(len(cuisines), dim))
    cuisine_emb /= np.linalg.norm(cuisine_emb, axis=1, keepdims=True)

    # merchants cluster in the centre and along a few high streets
    nm = m["n"]
    centre = rng.random(nm) < 0.55
    mx = np.where(centre, side / 2 + rng.normal(0, side / 6, nm), rng.uniform(0, side, nm)).clip(0, side - 1e-6)
    my = np.where(centre, side / 2 + rng.normal(0, side / 6, nm), rng.uniform(0, side, nm)).clip(0, side - 1e-6)
    zone = (mx // city["zone_km"]).astype(int) * city["grid"] + (my // city["zone_km"]).astype(int)
    cuisine = rng.choice(len(cuisines), size=nm, p=cw)
    price = rng.choice([1, 2, 3], size=nm, p=[0.4, 0.42, 0.18])
    quality = rng.normal(0, m["quality_sd"], nm)
    flavour = cuisine_emb[cuisine] + 0.35 * rng.normal(0, 1, size=(nm, dim))
    configured = np.exp(np.log(m["capacity_mean"]) + m["capacity_sigma"] * rng.normal(0, 1, nm))
    true_capacity = configured * np.exp(0.25 * rng.normal(0, 1, nm))
    prep = rng.uniform(*m["prep_minutes"], nm)
    commission = rng.uniform(*m["commission"], nm)
    promoted = rng.random(nm) < m["promoted_share"]
    history = cfg["time"]["history_days"]
    is_new = rng.random(nm) < m["new_share"]
    joined = np.where(is_new, rng.integers(history - 10, history + 4, nm), -365)
    basket = np.array(m["basket_eur"])[price - 1] * np.exp(0.15 * rng.normal(0, 1, nm))

    nc = c["n"]
    cx = rng.uniform(0, side, nc)
    cy = rng.uniform(0, side, nc)
    czone = (cx // city["zone_km"]).astype(int) * city["grid"] + (cy // city["zone_km"]).astype(int)
    mix = rng.dirichlet(c["cuisine_concentration"] * cw * len(cuisines), size=nc)
    taste = 1.6 * mix @ cuisine_emb + 0.25 * rng.normal(0, 1, size=(nc, dim))
    rate = c["sessions_per_week"] / 7 * np.exp(0.6 * rng.normal(0, 1, nc) - 0.18)

    def lognormal(spec):
        return np.exp(np.log(spec[0]) + spec[1] * rng.normal(0, 1, nc))

    truth = Truth(
        taste=taste,
        flavour=flavour,
        quality=quality,
        price_sens=lognormal(c["price_sensitivity"]),
        eta_sens=lognormal(c["eta_sensitivity"]),
        novelty=rng.beta(*c["novelty"], nc),
        loyalty=rng.beta(*c["loyalty"], nc),
        true_capacity=true_capacity,
    )
    catalog = Catalog(
        nm, mx, my, zone, cuisine, list(cuisines), price, prep, configured, commission, promoted, joined, basket
    )
    customers = Customers(nc, cx, cy, czone, rate)
    return City(catalog, customers, truth, seed, cfg)
