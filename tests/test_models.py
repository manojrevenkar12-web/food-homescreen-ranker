import numpy as np

from fhr.models.ranker import ndcg_at
from fhr.models.sequential import count_scores
from fhr.serve.bundle import load, save


def test_training_report_is_complete(small):
    *_, bundle = small
    r = bundle.report
    assert set(r) >= {"ndcg8", "retrieval", "sequential", "attention"}
    assert r["ndcg8"]["model"] > r["ndcg8"]["popularity"]
    assert 0 < r["attention"]["row_decay"] < 1 and 0 < r["attention"]["card_decay"] < 1


def test_retrieval_reaches_merchants_nobody_ordered_from(small):
    city, *_, bundle = small
    tt = bundle.two_tower
    unknown = np.flatnonzero(~tt.known)
    assert len(unknown) > 0
    s = tt.scores([], int(city.customers.zone[0]), 19, unknown)
    assert np.isfinite(s).all()


def test_count_scores_prefer_frequent_then_recent():
    s = count_scores([3, 1, 3, 2], 5)
    assert np.argmax(s) == 3
    assert s[2] > s[1]


def test_ndcg_perfect_and_reversed():
    import pandas as pd

    df = pd.DataFrame(
        {
            "sid": [1, 1, 1],
            "ordered": [True, False, False],
            "clicked": [True, True, False],
            "good": [3, 2, 1],
            "bad": [1, 2, 3],
        }
    )
    assert ndcg_at(df, "good") == 1.0
    assert ndcg_at(df, "bad") < 0.7


def test_bundle_round_trip_scores_identically(small, tmp_path):
    city, *_, bundle = small
    digest = save(bundle, tmp_path)
    assert digest.startswith("sha256:")
    again = load(tmp_path, city.catalog, city.config["city"]["grid"] ** 2)
    X = np.random.default_rng(0).normal(size=(20, 21))
    np.testing.assert_allclose(bundle.ranker.score(X), again.ranker.score(X))
    np.testing.assert_allclose(bundle.ranker.p_order(np.arange(3.0)), again.ranker.p_order(np.arange(3.0)))
    m = np.arange(10)
    np.testing.assert_allclose(
        bundle.two_tower.scores([1, 2], 0, 12, m), again.two_tower.scores([1, 2], 0, 12, m), atol=1e-6
    )
