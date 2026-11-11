# Repeatable tasks

## Add a ranking feature

1. Add it to `features.sql` and to `online()` in `src/fhr/features/build.py`, in the same position of `FEATURES`.
2. `pytest -q tests/test_features.py` (DuckDB against serving) and `pytest -q -m spark` (Spark against DuckDB).
3. `python scripts/evaluate.py`, then `python scripts/report.py`.

## Try a new row or a new objective on the page

1. Add the row to `PageAssembler.library` or the term to `score()` behind a flag, with its weight in `config/platform.toml`.
2. Add the variant to `scripts/evaluate.py` next to the ablations; compare on the same simulated week.
3. Before shipping: `python scripts/experiments.py`, `python scripts/gate.py`.

## Release a bundle

`python scripts/train_bundle.py OUT --world world --seed 11` writes the models and a `DIGEST`; the container
serves it from `/opt/ml/model` (SageMaker contract: `/ping`, `/invocations`).
