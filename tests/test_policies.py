from fhr.config import platform
from fhr.evals.online import eval_sessions
from fhr.policies.learned import PageAssembler, Reranked
from fhr.world.sim import MAX_CARDS, MAX_ROWS


def pages(small, policy, n=60):
    city, world, _, _ = small
    out = []
    for s in eval_sessions(city, 5).head(n).itertuples():
        ctx = world.context(int(s.sid), int(s.day), int(s.minute), int(s.customer))
        out.append((ctx, policy(ctx, world.obs)))
    return out


def test_page_shows_each_merchant_once_and_only_those_that_deliver(small):
    city, world, _, bundle = small
    for ctx, page in pages(small, PageAssembler(city.catalog, bundle)):
        shown = [m for r in page for m in r.merchants]
        assert len(shown) == len(set(shown))
        assert set(shown) <= set(ctx.available.tolist())
        assert len(page) <= MAX_ROWS and all(len(r.merchants) <= MAX_CARDS for r in page)


def test_guardrails_hold(small):
    city, world, _, bundle = small
    cfg = platform()["page"]
    new_days = platform()["policy"]["new_days"]
    for ctx, page in pages(small, PageAssembler(city.catalog, bundle)):
        a = ctx.available
        if city.catalog.promoted[a].sum() >= cfg["promoted_min_top3"] and len(page) >= 2:
            assert sum(city.catalog.promoted[m] for r in page[:3] for m in r.merchants) >= cfg["promoted_min_top3"]
        new_near = (ctx.day - city.catalog.joined_day[a] < new_days).sum()
        if new_near:
            shown_new = sum(1 for r in page for m in r.merchants if ctx.day - city.catalog.joined_day[m] < new_days)
            assert shown_new >= min(new_near, 1)


def test_congestion_discount_moves_full_kitchens_down(small):
    import numpy as np

    from fhr.policies.learned import score

    city, world, _, bundle = small
    ctx, _ = pages(small, Reranked(city.catalog, bundle), n=1)[0]
    calm = score(ctx, world.obs, city.catalog, bundle, gmv=False, explore=False)
    ctx.load[:] = 3.0
    busy = score(ctx, world.obs, city.catalog, bundle, gmv=False, explore=False)
    assert np.all(busy.value < calm.value)
