import numpy as np
import pandas as pd

from fhr.evals.experiments import _hash, analyse_customer_split, analyse_switchback
from fhr.world.sim import Logs


def test_assignment_is_deterministic_and_balanced():
    a = [_hash("s", c) < 0.5 for c in range(4000)]
    assert a == [_hash("s", c) < 0.5 for c in range(4000)]
    assert 0.47 < np.mean(a) < 0.53


def _logs(rows):
    s = pd.DataFrame(rows, columns=["sid", "day", "hour", "minute", "customer", "policy", "explore", "ordered", "via"])
    return Logs(s, pd.DataFrame(), pd.DataFrame())


def test_customer_split_recovers_a_planted_effect():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(3000):
        arm = "treatment" if c % 2 else "control"
        for k in range(3):
            p = 0.5 if arm == "treatment" else 0.4
            rows.append((c * 10 + k, 0, 12, 720, c, arm, False, bool(rng.random() < p), None))
    res = analyse_customer_split(_logs(rows), pd.DataFrame({"orders": [], "sessions": []}), reps=200, cuped=False)
    assert res["lo"] < 0.1 < res["hi"]


def test_cuped_narrows_the_interval_and_keeps_the_effect():
    rng = np.random.default_rng(1)
    rows, pre = [], []
    for c in range(3000):
        arm = "treatment" if c % 2 else "control"
        base = rng.uniform(0.1, 0.8)  # customers differ a lot in how often they order
        pre.append((c, rng.binomial(40, base), 40))
        for k in range(int(rng.integers(1, 6))):
            rows.append(
                (c * 10 + k, 0, 12, 720, c, arm, False, bool(rng.random() < base + 0.05 * (arm == "treatment")), None)
            )
    pre = pd.DataFrame(pre, columns=["customer", "orders", "sessions"]).set_index("customer")
    plain = analyse_customer_split(_logs(rows), pre, reps=300, cuped=False)
    adjusted = analyse_customer_split(_logs(rows), pre, reps=300)
    assert adjusted["hi"] - adjusted["lo"] < 0.9 * (plain["hi"] - plain["lo"])
    assert adjusted["lo"] < 0.05 < adjusted["hi"]


def test_switchback_drops_the_burn_in():
    rows = []
    for b in range(8):
        arm = "treatment" if b % 2 else "control"
        for m in range(0, 180, 2):
            minute = b * 180 + m
            ordered = arm == "treatment" if m >= 60 else True  # the first hour is contaminated
            rows.append((minute, minute // 1440, (minute // 60) % 24, minute, minute, arm, False, ordered, None))
    res = analyse_switchback(_logs(rows), block_hours=3, reps=100)
    assert res["diff"] == 1.0
