"""Home-screen service. Speaks the SageMaker inference contract (GET /ping, POST /invocations) and a plain
POST /v1/home. If the learned page cannot be built within the deadline, it serves today's page instead and
counts the fallback.

State: in production the customer and merchant counters come from the online feature store and the
catalog service. Here a `Replica` holds the same counters for a simulated city after its logging period.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel

from fhr.config import platform
from fhr.policies.heuristic import Heuristic
from fhr.policies.learned import PageAssembler
from fhr.train import Bundle
from fhr.world.city import City
from fhr.world.sim import World


@dataclass
class Replica:
    city: City
    world: World
    bundle: Bundle


class HomeRequest(BaseModel):
    customer: int
    minute: int


def create_app(replica: Replica, deadline_ms: float | None = None) -> FastAPI:
    deadline = deadline_ms if deadline_ms is not None else platform()["gate"]["max_p95_ms"] * 2
    reg = CollectorRegistry()
    latency = Histogram(
        "home_page_seconds",
        "Time to build a home page",
        ["policy"],
        registry=reg,
        buckets=(0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16),
    )
    served = Counter("home_pages_total", "Pages served", ["policy"], registry=reg)
    fallbacks = Counter("home_fallbacks_total", "Pages that fell back to today's screen", ["reason"], registry=reg)
    page = PageAssembler(replica.city.catalog, replica.bundle)
    current = Heuristic(replica.city.catalog)
    app = FastAPI(title="home screen", docs_url=None, redoc_url=None)

    def build(req: HomeRequest) -> dict:
        n = replica.city.customers.n
        if not 0 <= req.customer < n:
            raise HTTPException(404, "unknown customer")
        ctx = replica.world.context(sid=-1, day=req.minute // 1440, minute=req.minute, customer=req.customer)
        start = time.perf_counter()
        name = "page"
        try:
            rows = page(ctx, replica.world.obs)
        except Exception:  # noqa: BLE001 - any model failure must still produce a page
            fallbacks.labels("error").inc()
            rows, name = current(ctx, replica.world.obs), "current"
        took = (time.perf_counter() - start) * 1000
        if took > deadline and name == "page":
            fallbacks.labels("deadline").inc()
        latency.labels(name).observe(took / 1000)
        served.labels(name).inc()
        return {
            "customer": req.customer,
            "policy": name,
            "ms": round(took, 2),
            "rows": [{"kind": r.kind, "merchants": r.merchants} for r in rows],
        }

    @app.get("/ping")
    def ping() -> dict:
        return {"status": "ok"}

    @app.post("/invocations")
    def invocations(req: HomeRequest) -> dict:
        return build(req)

    @app.post("/v1/home")
    def home(req: HomeRequest) -> dict:
        return build(req)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:  # pragma: no cover - container entry point
    import uvicorn

    from fhr.evals.online import history
    from fhr.serve.bundle import load
    from fhr.world.city import build as build_city

    seed = int(os.environ.get("FHR_SEED", "11"))
    city = build_city(seed, os.environ.get("FHR_WORLD", "world"))
    world, _ = history(city, seed)
    bundle = load(
        __import__("pathlib").Path(os.environ.get("FHR_BUNDLE", "/opt/ml/model")),
        city.catalog,
        city.config["city"]["grid"] ** 2,
    )
    uvicorn.run(create_app(Replica(city, world, bundle)), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
