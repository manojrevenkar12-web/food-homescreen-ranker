import pytest

from fhr.evals.online import history
from fhr.train import train
from fhr.world.city import build


@pytest.fixture(scope="session")
def small():
    """A small city after its logging period, with models trained on it."""
    city = build(5, "world_small")
    world, logs = history(city, 5)
    c = city.config
    bundle = train(
        logs,
        city.catalog,
        city.customers,
        c["city"]["grid"] ** 2,
        c["time"]["history_days"],
        c["city"]["delivery_radius_km"],
    )
    return city, world, logs, bundle
