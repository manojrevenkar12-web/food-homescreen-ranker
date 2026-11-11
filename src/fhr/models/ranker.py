"""LightGBM LambdaRank over the candidates of a session, plus a calibration to P(order | the card is seen).

Training rows are viewed impressions (the app logs a card only when it is on screen), so the label is not
confounded by cards nobody scrolled to. Labels: ordered 3, clicked 1, otherwise 0.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fhr.config import platform
from fhr.features.build import FEATURES

COLUMNS = [*FEATURES, "tt_score"]


class Ranker:
    def __init__(self):
        cfg = platform()["ranker"]
        self.cfg = cfg
        self.model = lgb.LGBMRanker(
            objective=cfg["objective"],
            num_leaves=cfg["num_leaves"],
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            min_child_samples=cfg["min_child_samples"],
            random_state=0,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )
        self.calib = LogisticRegression()

    def labels(self, df: pd.DataFrame) -> np.ndarray:
        return np.where(df.ordered, self.cfg["label_order"], np.where(df.clicked, self.cfg["label_click"], 0))

    def fit(self, train: pd.DataFrame, calib: pd.DataFrame) -> Ranker:
        train = train.sort_values("sid", kind="stable")
        groups = train.groupby("sid", sort=False).size().to_numpy()
        self.model.fit(train[COLUMNS], self.labels(train), group=groups, categorical_feature=["cuisine"])
        self.calib.fit(self.score(calib)[:, None], calib.ordered.astype(int))
        return self

    def score(self, df: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Serving calls this with a numpy matrix in COLUMNS order: the booster directly, one thread, no pandas."""
        X = df[COLUMNS].to_numpy(dtype=float) if isinstance(df, pd.DataFrame) else np.asarray(df, dtype=float)
        return self.model.booster_.predict(X, num_threads=1)

    def p_order(self, scores: np.ndarray) -> np.ndarray:
        return self.calib.predict_proba(np.asarray(scores)[:, None])[:, 1]


def ndcg_at(df: pd.DataFrame, score_col: str, k: int = 8) -> float:
    """Mean NDCG@k over sessions with at least one order, graded 3/1/0."""
    out = []
    for _, g in df.groupby("sid"):
        if not g.ordered.any():
            continue
        rel = np.where(g.ordered, 3, np.where(g.clicked, 1, 0))
        order = np.argsort(-g[score_col].to_numpy(), kind="stable")
        gains = (2.0 ** rel[order][:k] - 1) / np.log2(np.arange(2, min(k, len(g)) + 2))
        ideal = (2.0 ** np.sort(rel)[::-1][:k] - 1) / np.log2(np.arange(2, min(k, len(g)) + 2))
        out.append(gains.sum() / ideal.sum())
    return float(np.mean(out)) if out else float("nan")
