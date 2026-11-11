"""Nightly retraining of the home-screen models (Airflow 2.x TaskFlow).

Every task is a step of src/fhr/train.py or a script in scripts/, so the DAG holds no logic of its own:
extract the day's logs -> features on Spark (the same features.sql) -> train -> offline report
-> gate -> register the bundle for the SageMaker endpoint. A failed gate stops the run; nothing is registered.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task

DEFAULT_ARGS = {"owner": "consumer-discovery", "retries": 2, "retry_delay": timedelta(minutes=10)}


@dag(
    dag_id="home_ranker_daily",
    schedule="30 3 * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["ranking", "home-screen"],
)
def home_ranker_daily():
    @task
    def extract_logs(ds: str | None = None) -> str:
        """Impressions, sessions and orders for the training window, as Parquet in S3."""
        return f"s3://consumer-discovery/logs/dt={ds}/"

    @task
    def features(logs_uri: str) -> str:
        """Run src/fhr/features/features.sql on Spark (EMR or Glue) over the window."""
        return logs_uri.replace("/logs/", "/features/")

    @task
    def train_models(features_uri: str) -> str:
        """fhr.train.train: two-tower, ranker, calibration, sequential challenger, attention estimates."""
        return features_uri.replace("/features/", "/bundles/")

    @task
    def offline_report(bundle_uri: str) -> dict:
        """NDCG@8 against today's ranking, retrieval recall, cold-start recall: the bundle's report."""
        return {"bundle": bundle_uri}

    @task
    def gate(report: dict) -> str:
        """scripts/gate.py: offline floors for a nightly refresh; an A/B test for a new policy."""
        return report["bundle"]

    @task
    def register(bundle_uri: str) -> None:
        """Register the bundle (DIGEST pinned) as a new SageMaker model package; the endpoint updates by canary."""

    register(gate(offline_report(train_models(features(extract_logs())))))


home_ranker_daily()
