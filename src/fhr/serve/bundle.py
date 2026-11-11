"""Save and load a trained bundle without pickle: torch state dicts, LightGBM text, JSON for the rest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

from fhr.models.ranker import Ranker
from fhr.models.sequential import Sequential
from fhr.models.two_tower import TwoTower
from fhr.train import Bundle
from fhr.world.city import Catalog


def save(bundle: Bundle, path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"m": bundle.two_tower.m_tower.state_dict(), "c": bundle.two_tower.c_tower.state_dict()}, path / "two_tower.pt"
    )
    if bundle.sequential is not None:
        torch.save(bundle.sequential.model.state_dict(), path / "sequential.pt")
    (path / "ranker.txt").write_text(bundle.ranker.model.booster_.model_to_string(), encoding="utf-8")
    meta = {
        "row_decay": bundle.row_decay,
        "card_decay": bundle.card_decay,
        "known": np.flatnonzero(bundle.two_tower.known).tolist(),
        "calib": {"coef": bundle.ranker.calib.coef_.tolist(), "intercept": bundle.ranker.calib.intercept_.tolist()},
        "sequential": bundle.sequential is not None,
        "report": bundle.report,
    }
    (path / "meta.json").write_text(json.dumps(meta, default=float), encoding="utf-8")
    h = hashlib.sha256()
    for f in sorted(path.glob("*")):
        if f.name != "DIGEST":
            h.update(f.name.encode() + f.read_bytes())
    digest = "sha256:" + h.hexdigest()
    (path / "DIGEST").write_text(digest)
    return digest


def load(path: Path, catalog: Catalog, n_zones: int) -> Bundle:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    tt = TwoTower(catalog, n_zones)
    state = torch.load(path / "two_tower.pt", weights_only=True)
    tt.m_tower.load_state_dict(state["m"])
    tt.c_tower.load_state_dict(state["c"])
    tt.m_tower.eval()
    tt.c_tower.eval()
    tt.known = np.zeros(catalog.n, dtype=bool)
    tt.known[meta["known"]] = True
    ranker = Ranker()
    booster = lgb.Booster(model_str=(path / "ranker.txt").read_text(encoding="utf-8"))
    ranker.model._Booster = booster
    ranker.model.fitted_ = True
    ranker.calib.classes_ = np.array([0, 1])
    ranker.calib.coef_ = np.array(meta["calib"]["coef"])
    ranker.calib.intercept_ = np.array(meta["calib"]["intercept"])
    seq = None
    if meta["sequential"]:
        seq = Sequential(catalog.n)
        seq.model.load_state_dict(torch.load(path / "sequential.pt", weights_only=True))
        seq.model.eval()
    return Bundle(tt, ranker, seq, meta["row_decay"], meta["card_decay"], meta["report"])
