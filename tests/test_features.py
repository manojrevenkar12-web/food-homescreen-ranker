import copy

import numpy as np

from fhr.features.build import FEATURES, offline, online
from fhr.policies.heuristic import Heuristic
from fhr.world.city import build
from fhr.world.sim import World, schedule


def test_sql_features_equal_serving_features():
    """The ranker is trained on features.sql and served on online(): they must agree, point in time."""
    city = build(3, "world_small")
    w = World(city)
    sessions = schedule(city, 0, 4, 3)
    probe = set(sessions[sessions.day == 3].sid.sample(25, random_state=0))
    snap = {}
    step = w.step

    def capture(ctx, page, explore, policy, seed):
        if ctx.sid in probe:
            snap[ctx.sid] = (ctx, copy.deepcopy(w.obs))
        return step(ctx, page, explore, policy, seed)

    w.step = capture
    h = Heuristic(city.catalog)
    logs = w.run(sessions, {"h": h}, lambda c: "h", 3)
    feats = offline(logs, city.catalog)
    checked = 0
    for sid, (ctx, obs) in snap.items():
        rows = feats[feats.sid == sid]
        if rows.empty:
            continue
        on = online(ctx, obs, city.catalog, rows.merchant.to_numpy())
        np.testing.assert_allclose(on, rows[FEATURES].to_numpy(dtype=float), atol=1e-9)
        checked += 1
    assert checked >= 10


def test_merchant_counters_exclude_today(small):
    city, world, logs, _ = small
    feats = offline(logs, city.catalog)
    first_day = feats[feats.day == 0]
    assert (first_day.m_orders == 0).all()
