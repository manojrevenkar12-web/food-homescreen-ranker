import numpy as np

from fhr.evals.online import eval_sessions, metrics
from fhr.policies.heuristic import Heuristic
from fhr.world.sim import Row


def test_same_policy_same_future(small):
    city, world, _, _ = small
    s = eval_sessions(city, 5).head(400)
    h = Heuristic(city.catalog)
    a = world.fork().run(s, {"h": h}, lambda c: "h", 9)
    b = world.fork().run(s, {"h": h}, lambda c: "h", 9)
    assert a.orders.equals(b.orders)
    assert a.impressions.equals(b.impressions)


def test_fork_does_not_touch_the_original(small):
    city, world, _, _ = small
    before = world.obs.orders_cm.sum()
    s = eval_sessions(city, 5).head(200)
    world.fork().run(s, {"h": Heuristic(city.catalog)}, lambda c: "h", 9)
    assert world.obs.orders_cm.sum() == before


def test_kitchens_slow_down_under_load(small):
    _, world, _, _ = small
    delay = world._delay(np.array([0.5, 0.7, 1.4, 2.1]))
    assert delay[0] == 0 and delay[1] == 0
    assert 25 < delay[2] < 27 and delay[3] > delay[2]


def test_cards_from_merchants_that_do_not_deliver_are_not_seen(small):
    city, world, _, _ = small
    s = eval_sessions(city, 5).iloc[0]
    ctx = world.context(int(s.sid), int(s.day), int(s.minute), int(s.customer))
    outside = sorted(set(range(city.catalog.n)) - set(ctx.available.tolist()))[:5]
    w = world.fork()
    w.step(ctx, [Row("x", outside)], False, "p", 1)
    assert w.logs().impressions.empty


def test_metrics_are_rates(small):
    city, world, _, _ = small
    s = eval_sessions(city, 5).head(300)
    m = metrics(world.fork().run(s, {"h": Heuristic(city.catalog)}, lambda c: "h", 9), city)
    assert 0 < m["orders_per_session"] < 1
    assert 0 <= m["late_share"] <= 1 and 0 <= m["top10_share"] <= 1
