"""A GRU over a customer's order sequence that predicts the next merchant.

It competes with the count-based "order again" row (most-ordered first). It ships only if it beats that row
by `sequential.min_gain_recall10` on next-order recall@10: a sequence model has to earn its complexity.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fhr.config import platform


class GRURec(nn.Module):
    def __init__(self, n_merchants: int, dim: int):
        super().__init__()
        self.pad = n_merchants
        self.emb = nn.Embedding(n_merchants + 1, dim, padding_idx=n_merchants)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.out = nn.Linear(dim, n_merchants)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.emb(seq))
        return self.out(h[:, -1])


class Sequential:
    def __init__(self, n_merchants: int):
        self.cfg = platform()["sequential"]
        torch.manual_seed(0)
        self.n = n_merchants
        self.model = GRURec(n_merchants, self.cfg["dim"])

    def _pad(self, hist: list[int]) -> np.ndarray:
        L = self.cfg["history_len"]
        h = hist[-L:]
        out = np.full(L, self.n, dtype=np.int64)
        out[L - len(h) :] = h  # left padding: the last step is the latest order
        return out

    def fit(self, histories: list[list[int]], targets: list[int]) -> Sequential:
        cfg = self.cfg
        rng = np.random.default_rng(0)
        x = torch.as_tensor(np.stack([self._pad(h) for h in histories]))
        y = torch.as_tensor(np.array(targets))
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"])
        for _ in range(cfg["epochs"]):
            order = rng.permutation(len(y))
            for i in range(0, len(y), cfg["batch"]):
                b = order[i : i + cfg["batch"]]
                loss = nn.functional.cross_entropy(self.model(x[b]), y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def scores(self, hist: list[int]) -> np.ndarray:
        return self.model(torch.as_tensor(self._pad(hist)[None]))[0].numpy()


def count_scores(hist: list[int], n: int) -> np.ndarray:
    """The current order-again row: most-ordered first, most recent breaks ties."""
    s = np.zeros(n)
    for k, m in enumerate(hist):
        s[m] += 1 + 1e-3 * k
    return s
