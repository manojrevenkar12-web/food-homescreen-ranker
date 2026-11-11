"""A ceiling: the page a platform would build if it knew every customer's taste. Reads the truth."""

from __future__ import annotations

import numpy as np

from fhr.world.sim import MAX_CARDS, MAX_ROWS, Context, Observed, Row, World


class Oracle:
    name = "oracle"

    def __init__(self, world: World):
        self.world = world

    def __call__(self, ctx: Context, obs: Observed) -> list[Row]:
        w = self.world
        rng = np.random.default_rng(0)
        # the deterministic part of the utility: the session noise averaged out
        saved = w.city.config["behaviour"]["noise"]
        w.city.config["behaviour"]["noise"] = 0.0
        u = w._utility(ctx, rng)
        w.city.config["behaviour"]["noise"] = saved
        top = ctx.available[np.argsort(-u)][: MAX_ROWS * MAX_CARDS]
        return [Row("oracle", [int(m) for m in top[i : i + MAX_CARDS]]) for i in range(0, len(top), MAX_CARDS)]
