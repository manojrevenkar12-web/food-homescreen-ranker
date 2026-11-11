"""Two-tower retrieval: a customer tower and a merchant tower trained on orders with in-batch negatives.

- logQ correction: popular merchants appear as in-batch negatives more often; their logit is corrected by
  the log of their sampling frequency so the model does not learn "popular = bad".
- Cold start: the merchant id embedding is dropped for a share of training rows, so the content path
  (cuisine, price, zone) learns to place a merchant nobody has ordered from yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from fhr.config import platform
from fhr.world.city import Catalog

N_HOURS = 4  # breakfast, lunch, afternoon, dinner+late


def hour_bucket(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h)
    return np.select([h < 11, h < 15, h < 17], [0, 1, 2], 3)


class MerchantTower(nn.Module):
    def __init__(self, n_merchants: int, n_cuisines: int, n_zones: int, dim: int):
        super().__init__()
        self.unknown = n_merchants
        self.id = nn.Embedding(n_merchants + 1, dim)
        self.cuisine = nn.Embedding(n_cuisines, dim)
        self.price = nn.Embedding(4, dim)
        self.zone = nn.Embedding(n_zones, dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, ids, cuisine, price, zone):
        return self.out(self.id(ids) + self.cuisine(cuisine) + self.price(price) + self.zone(zone))


class CustomerTower(nn.Module):
    def __init__(self, n_merchants: int, n_cuisines: int, n_zones: int, dim: int):
        super().__init__()
        self.hist = nn.EmbeddingBag(n_merchants + 1, dim, mode="mean", padding_idx=n_merchants)
        self.cuisine_mix = nn.Linear(n_cuisines, dim)
        self.zone = nn.Embedding(n_zones, dim)
        self.hour = nn.Embedding(N_HOURS, dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, hist, mix, zone, hour):
        return self.out(self.hist(hist) + self.cuisine_mix(mix) + self.zone(zone) + self.hour(hour))


@dataclass
class CustomerState:
    hist: np.ndarray  # padded merchant ids
    mix: np.ndarray  # cuisine shares of past orders


class TwoTower:
    def __init__(self, catalog: Catalog, n_zones: int):
        cfg = platform()["two_tower"]
        self.catalog = catalog
        self.cfg = cfg
        torch.manual_seed(0)
        nc = len(catalog.cuisines)
        self.m_tower = MerchantTower(catalog.n, nc, n_zones, cfg["dim"])
        self.c_tower = CustomerTower(catalog.n, nc, n_zones, cfg["dim"])
        self._m_emb: np.ndarray | None = None
        self.known: np.ndarray = np.zeros(catalog.n, dtype=bool)

    # ------------------------------------------------------------------ inputs
    def customer_state(self, history: list[int]) -> CustomerState:
        L = self.cfg["history_len"]
        h = history[-L:]
        hist = np.full(L, self.catalog.n, dtype=np.int64)
        hist[: len(h)] = h
        mix = np.zeros(len(self.catalog.cuisines), dtype=np.float32)
        if h:
            np.add.at(mix, self.catalog.cuisine[h], 1.0)
            mix /= mix.sum()
        return CustomerState(hist, mix)

    def _merchant_inputs(self, ids: np.ndarray, drop_unknown: bool = True):
        cat = self.catalog
        id_in = np.where(self.known[ids] | (not drop_unknown), ids, cat.n)
        return (
            torch.as_tensor(id_in),
            torch.as_tensor(cat.cuisine[ids]),
            torch.as_tensor(cat.price[ids]),
            torch.as_tensor(cat.zone[ids]),
        )

    # ------------------------------------------------------------------ training
    def fit(self, examples: pd.DataFrame) -> TwoTower:
        """examples: one row per order with columns merchant, zone, hour, history (list of earlier merchants)."""
        cfg = self.cfg
        rng = np.random.default_rng(0)
        self.known = np.zeros(self.catalog.n, dtype=bool)
        self.known[examples.merchant.unique()] = True
        states = [self.customer_state(list(h)) for h in examples.history]
        hist = torch.as_tensor(np.stack([s.hist for s in states]))
        mix = torch.as_tensor(np.stack([s.mix for s in states]))
        zone = torch.as_tensor(examples.zone.to_numpy().copy())
        hour = torch.as_tensor(hour_bucket(examples.hour.to_numpy()))
        target = examples.merchant.to_numpy()
        freq = np.bincount(target, minlength=self.catalog.n) + 1.0
        logq = torch.as_tensor(np.log(freq / freq.sum()), dtype=torch.float32)
        params = list(self.m_tower.parameters()) + list(self.c_tower.parameters())
        opt = torch.optim.Adam(params, lr=cfg["lr"])
        n = len(examples)
        for _ in range(cfg["epochs"]):
            order = rng.permutation(n)
            for i in range(0, n, cfg["batch"]):
                b = order[i : i + cfg["batch"]]
                ids = target[b]
                m_ids = np.where(rng.random(len(b)) < cfg["id_dropout"], self.catalog.n, ids)
                cat = self.catalog
                m = self.m_tower(
                    torch.as_tensor(m_ids),
                    torch.as_tensor(cat.cuisine[ids]),
                    torch.as_tensor(cat.price[ids]),
                    torch.as_tensor(cat.zone[ids]),
                )
                c = self.c_tower(hist[b], mix[b], zone[b], hour[b])
                logits = (nn.functional.normalize(c, dim=1) @ nn.functional.normalize(m, dim=1).T) / cfg["temperature"]
                logits = logits - logq[torch.as_tensor(ids)][None, :]
                same = torch.as_tensor(ids[:, None] == ids[None, :]) & ~torch.eye(len(b), dtype=torch.bool)
                logits = logits.masked_fill(same, -1e9)
                loss = nn.functional.cross_entropy(logits, torch.arange(len(b)))
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.m_tower.eval()
        self.c_tower.eval()
        self._m_emb = None
        return self

    # ------------------------------------------------------------------ serving
    @torch.no_grad()
    def merchant_embeddings(self) -> np.ndarray:
        if self._m_emb is None:
            ids = np.arange(self.catalog.n)
            e = self.m_tower(*self._merchant_inputs(ids))
            self._m_emb = nn.functional.normalize(e, dim=1).numpy()
        return self._m_emb

    @torch.no_grad()
    def customer_embedding(self, history: list[int], zone: int, hour: int) -> np.ndarray:
        s = self.customer_state(history)
        e = self.c_tower(
            torch.as_tensor(s.hist[None]),
            torch.as_tensor(s.mix[None]),
            torch.as_tensor([zone]),
            torch.as_tensor(hour_bucket(np.array([hour]))),
        )
        return nn.functional.normalize(e, dim=1).numpy()[0]

    def scores(self, history: list[int], zone: int, hour: int, merchants: np.ndarray) -> np.ndarray:
        return self.merchant_embeddings()[merchants] @ self.customer_embedding(history, zone, hour)


def examples_from_logs(logs, customers_zone: np.ndarray) -> pd.DataFrame:
    """One row per order with the customer's earlier orders as history (point in time)."""
    o = logs.orders.merge(logs.sessions[["sid", "hour", "minute"]], on="sid").sort_values("sid")
    hist: dict[int, list[int]] = {}
    rows = []
    for r in o.itertuples(index=False):
        h = hist.setdefault(int(r.customer), [])
        rows.append(
            (int(r.sid), int(r.customer), int(r.merchant), int(customers_zone[r.customer]), int(r.hour), list(h))
        )
        h.append(int(r.merchant))
    return pd.DataFrame(rows, columns=["sid", "customer", "merchant", "zone", "hour", "history"])
