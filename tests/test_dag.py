"""The DAG file parses and its chain matches the training pipeline (Airflow itself runs in CI job "airflow")."""

import ast

from fhr.config import ROOT


def test_dag_tasks_in_order():
    tree = ast.parse((ROOT / "dags" / "home_ranker_daily.py").read_text())
    tasks = [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and any(isinstance(d, ast.Name) and d.id == "task" for d in n.decorator_list)
    ]
    assert tasks == ["extract_logs", "features", "train_models", "offline_report", "gate", "register"]
