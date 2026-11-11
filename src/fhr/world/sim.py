"""Sessions on the home screen. THE TRUTH decides what a customer looks at, clicks and orders.

A policy sees a `Context` (the customer, the time, the merchants that deliver there with the platform's own
ETA estimate and live kitchen load) and `Observed` (the order history the platform logged). It returns a page:
rows of merchants. The customer scrolls down rows and swipes along them with falling attention, clicks
what looks good, orders the best clicked merchant if it beats cooking at home, and otherwise sometimes
searches. Every order adds load to a kitchen; a kitchen past its real capacity gets slower for everyone.

Random numbers are drawn per session and per merchant, so two policies run on the same city see the same
customers, the same moods and the same scrolling: their difference is the policy's alone.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fhr.world.city import City

MAX_ROWS = 6
MAX_CARDS = 8


@dataclass
class Row:
    kind: str
    merchants: list[int]


@dataclass
class Context:
    sid: int
    day: int
    hour: int
    minute: int  # minutes since day 0, 00:00
    customer: int
    zone: int
    available: np.ndarray  # merchant ids that deliver here and are on the platform
    distance: np.ndarray  # km, aligned with available
    eta: np.ndarray  # the platform's ETA estimate in minutes, aligned with available
    load: np.ndarray  # orders in the last hour / configured capacity, aligned with available


@dataclass
class Observed:
    """The platform's own data: what was ordered, when, how late. Policies may read all of it."""

    orders_cm: np.ndarray  # customer x merchant order counts
    last_day_cm: np.ndarray  # customer x merchant: day of the last order, -1 if none
    daily_orders: np.ndarray  # day x merchant (merchant counters are batch: read up to yesterday)
    daily_late: np.ndarray
    daily_stars_sum: np.ndarray
    daily_stars_n: np.ndarray
    daily_impr: np.ndarray
    daily_clicks: np.ndarray
    last_orders: dict[int, list[int]] = field(default_factory=dict)  # customer -> merchant sequence


Policy = Callable[[Context, Observed], list[Row]]


@dataclass
class Logs:
    sessions: pd.DataFrame
    impressions: pd.DataFrame
    orders: pd.DataFrame


class World:
    def __init__(self, city: City):
        self.city = city
        cat = city.catalog
        n_c, n_m = city.customers.n, cat.n
        days = city.config["time"]["history_days"] + city.config["time"]["eval_days"] + 1

        def per_day():
            return np.zeros((days, n_m), dtype=np.float64)

        self.obs = Observed(
            orders_cm=np.zeros((n_c, n_m), dtype=np.int16),
            last_day_cm=np.full((n_c, n_m), -1, dtype=np.int16),
            daily_orders=per_day(),
            daily_late=per_day(),
            daily_stars_sum=per_day(),
            daily_stars_n=per_day(),
            daily_impr=per_day(),
            daily_clicks=per_day(),
        )
        self.late_cm = np.zeros((n_c, n_m), dtype=np.float32)
        self.recent: list[deque] = [deque() for _ in range(n_m)]
        dx = city.customers.x[:, None] - cat.x[None, :]
        dy = city.customers.y[:, None] - cat.y[None, :]
        self.dist = np.sqrt(dx**2 + dy**2).astype(np.float32)
        self._sess: list = []
        self._imp: list = []
        self._ord: list = []

    def fork(self) -> World:
        """A copy at this point in time, for running another policy on the same future."""
        w = copy.copy(self)
        w.obs = copy.deepcopy(self.obs)
        w.late_cm = self.late_cm.copy()
        w.recent = [deque(d) for d in self.recent]
        w._sess, w._imp, w._ord = [], [], []
        return w

    # ------------------------------------------------------------------ kitchen load and delivery time
    def _load(self, ms: np.ndarray, minute: int, capacity: np.ndarray) -> np.ndarray:
        out = np.empty(len(ms))
        for i, m in enumerate(ms):
            q = self.recent[m]
            while q and q[0] <= minute - 60:
                q.popleft()
            out[i] = len(q)
        return out / capacity[ms]

    def _delay(self, load: np.ndarray) -> np.ndarray:
        c = self.city.config["congestion"]
        over = np.clip((load - c["start_load"]) / c["start_load"], 0, None)
        return c["minutes"] * over ** c["power"]

    def context(self, sid: int, day: int, minute: int, customer: int) -> Context:
        cat = self.city.catalog
        radius = self.city.config["city"]["delivery_radius_km"]
        d = self.dist[customer]
        avail = np.flatnonzero((d <= radius) & (cat.joined_day <= day))
        load = self._load(avail, minute, cat.capacity)
        cw = self.city.config["congestion"]["courier_wait"]
        eta = cat.prep[avail] + 3.0 * d[avail] + (cw[0] + cw[1]) / 2 + self._delay(load)
        return Context(
            sid,
            day,
            (minute // 60) % 24,
            minute,
            customer,
            int(self.city.customers.zone[customer]),
            avail,
            d[avail],
            eta,
            load,
        )

    # ------------------------------------------------------------------ the customer (truth)
    def _utility(self, ctx: Context, rng: np.random.Generator) -> np.ndarray:
        t, cat, b = self.city.truth, self.city.catalog, self.city.config["behaviour"]
        c, a = ctx.customer, ctx.available
        noise = rng.gumbel(0, b["noise"], cat.n)[a]
        prev = self.obs.orders_cm[c, a].astype(float)
        breakfast = np.isin(np.array(cat.cuisines)[cat.cuisine[a]], self.city.config["merchants"]["breakfast"])
        time_fit = np.where(ctx.hour < 11, np.where(breakfast, 0.8, -0.6), np.where(breakfast, -0.3, 0.0))
        return (
            0.9 * t.taste[c] @ t.flavour[a].T
            + b["quality_weight"] * t.quality[a]
            - t.price_sens[c] * 0.8 * (cat.price[a] - 2)
            - t.eta_sens[c] * (ctx.eta - 30) / 10
            - b["distance_weight"] * ctx.distance
            + t.loyalty[c] * 1.2 * np.log1p(prev)
            + t.novelty[c] * 0.5 * (prev == 0)
            - b["late_penalty"] * self.late_cm[c, a]
            + time_fit
            + noise
        )

    def step(self, ctx: Context, page: list[Row], explore: bool, policy: str, seed: int) -> int | None:
        """One session. Returns the merchant ordered from, or None."""
        city, b, cat = self.city, self.city.config["behaviour"], self.city.catalog
        rng = np.random.default_rng([seed, ctx.sid])
        u_all = np.full(cat.n, -np.inf)
        u_all[ctx.available] = self._utility(ctx, rng)
        scroll = rng.random(MAX_ROWS)
        swipe = rng.random((MAX_ROWS, MAX_CARDS))
        click_draw = rng.random(cat.n)
        search_draw = rng.random()
        eta_noise = rng.normal(0, 3.0, cat.n)
        rating_draw = rng.random()
        avail_pos = {int(m): i for i, m in enumerate(ctx.available)}

        seen: set[int] = set()
        clicked: list[int] = []
        viewed: list[tuple] = []
        for r, row in enumerate(page[:MAX_ROWS]):
            if r > 0 and scroll[r] > b["row_attention"]:
                break
            for j, m in enumerate(row.merchants[:MAX_CARDS]):
                if j >= b["visible_cards"] and swipe[r, j] > b["card_attention"]:
                    break
                if m in seen or m not in avail_pos:
                    continue
                seen.add(m)
                p_click = 1 / (1 + np.exp(-2.0 * (u_all[m] - b["click_threshold"])))
                click = bool(click_draw[m] < p_click)
                if click:
                    clicked.append(m)
                k = avail_pos[m]
                viewed.append(
                    (
                        ctx.sid,
                        ctx.customer,
                        m,
                        r,
                        row.kind,
                        j,
                        explore,
                        click,
                        float(ctx.distance[k]),
                        float(ctx.eta[k]),
                        float(ctx.load[k]),
                    )
                )
        # impressions and clicks reach the platform's counters after the session, as the event stream would
        self._imp.extend(viewed)
        for v in viewed:
            self.obs.daily_impr[ctx.day, v[2]] += 1
            self.obs.daily_clicks[ctx.day, v[2]] += v[7]
        choice, via = None, None
        if clicked:
            best = max(clicked, key=lambda m: u_all[m])
            if u_all[best] > b["order_threshold"]:
                choice, via = best, "page"
        if choice is None and search_draw < b["search_rate"]:
            best = int(ctx.available[np.argmax(u_all[ctx.available])]) if len(ctx.available) else None
            if best is not None and u_all[best] > b["order_threshold"] + b["search_cost"]:
                choice, via = best, "search"
        if choice is not None:
            i = avail_pos[choice]
            true_load = self._load(np.array([choice]), ctx.minute, city.truth.true_capacity)[0]
            cw = city.config["congestion"]["courier_wait"]
            eta_true = (
                cat.prep[choice]
                + 3.0 * ctx.distance[i]
                + rng.uniform(*cw)
                + self._delay(np.array([true_load]))[0]
                + eta_noise[choice]
            )
            late = eta_true > b["late_minutes"]
            self.recent[choice].append(ctx.minute)
            o = self.obs
            o.orders_cm[ctx.customer, choice] += 1
            o.daily_orders[ctx.day, choice] += 1
            o.daily_late[ctx.day, choice] += late
            if late:
                self.late_cm[ctx.customer, choice] += 1
            stars = None
            if rating_draw < 0.3:  # three in ten customers rate
                stars = float(np.clip(3.6 + 0.7 * city.truth.quality[choice] - 1.2 * late + rng.normal(0, 0.6), 1, 5))
                o.daily_stars_sum[ctx.day, choice] += stars
                o.daily_stars_n[ctx.day, choice] += 1
            o.last_day_cm[ctx.customer, choice] = ctx.day
            o.last_orders.setdefault(ctx.customer, []).append(choice)
            basket = cat.basket[choice] * np.exp(rng.normal(0, 0.2))
            self._ord.append(
                (
                    ctx.sid,
                    ctx.customer,
                    choice,
                    via,
                    float(ctx.eta[i]),
                    float(eta_true),
                    bool(late),
                    float(basket),
                    float(basket * cat.commission[choice]),
                    bool(choice in seen),
                    stars,
                )
            )
        self._sess.append(
            (ctx.sid, ctx.day, ctx.hour, ctx.minute, ctx.customer, policy, explore, choice is not None, via)
        )
        return choice

    def run(
        self,
        sessions: pd.DataFrame,
        policies: dict[str, Policy],
        assign: Callable[[int], str],
        seed: int,
        explore: Callable[[int], bool] | None = None,
    ) -> Logs:
        for s in sessions.itertuples(index=False):
            ctx = self.context(int(s.sid), int(s.day), int(s.minute), int(s.customer))
            name = assign(int(s.customer))
            ex = bool(explore(int(s.sid))) if explore else False
            page = policies[name](ctx, self.obs) if not ex else policies[name + ":explore"](ctx, self.obs)
            self.step(ctx, page, ex, name, seed)
        return self.logs()

    def logs(self) -> Logs:
        return Logs(
            pd.DataFrame(
                self._sess, columns=["sid", "day", "hour", "minute", "customer", "policy", "explore", "ordered", "via"]
            ),
            pd.DataFrame(
                self._imp,
                columns=[
                    "sid",
                    "customer",
                    "merchant",
                    "row",
                    "row_kind",
                    "pos",
                    "explore",
                    "clicked",
                    "distance",
                    "eta",
                    "load",
                ],
            ),
            pd.DataFrame(
                self._ord,
                columns=[
                    "sid",
                    "customer",
                    "merchant",
                    "via",
                    "eta_shown",
                    "eta_true",
                    "late",
                    "basket",
                    "commission",
                    "from_page",
                    "stars",
                ],
            ),
        )


def schedule(city: City, day_from: int, day_to: int, seed: int) -> pd.DataFrame:
    """When each customer opens the app. The same schedule is replayed for every policy."""
    rng = np.random.default_rng([seed, 7919])
    w = np.array(city.config["time"]["hour_weights"], dtype=float)
    w /= w.sum()
    rows = []
    for day in range(day_from, day_to):
        k = rng.poisson(city.customers.rate)
        cust = np.repeat(np.arange(city.customers.n), k)
        hours = rng.choice(24, size=len(cust), p=w)
        minutes = day * 1440 + hours * 60 + rng.integers(0, 60, len(cust))
        rows.append(pd.DataFrame({"day": day, "minute": minutes, "customer": cust}))
    df = pd.concat(rows).sort_values("minute", kind="stable").reset_index(drop=True)
    df.insert(0, "sid", np.arange(len(df)) + day_from * 1_000_000)
    return df
