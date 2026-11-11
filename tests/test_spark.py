"""features.sql must give the same answer on Spark (the nightly job) as on DuckDB (development)."""

import numpy as np
import pytest

from fhr.features.build import FEATURES, offline, offline_spark

pytestmark = pytest.mark.spark


def test_duckdb_and_spark_agree(small):
    pyspark = pytest.importorskip("pyspark")
    spark = (
        pyspark.sql.SparkSession.builder.master("local[2]")
        .appName("features")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    city, _, logs, _ = small
    try:
        a = offline(logs, city.catalog).sort_values(["sid", "merchant"]).reset_index(drop=True)
        b = offline_spark(logs, city.catalog, spark).sort_values(["sid", "merchant"]).reset_index(drop=True)
    finally:
        spark.stop()
    assert len(a) == len(b)
    np.testing.assert_allclose(a[FEATURES].to_numpy(dtype=float), b[FEATURES].to_numpy(dtype=float), atol=1e-9)
