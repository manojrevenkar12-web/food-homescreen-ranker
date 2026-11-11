# Working in this repository

## Fast feedback

- `pytest -q` (about 20 s on the small city) and `ruff check . && ruff format --check .` before every commit.
- `pytest -q -m spark` when `src/fhr/features/features.sql` changes: Spark must agree with DuckDB.
- `cd web && npm test` for the dashboard.

## Invariants

- **Truth stays in the world.** Only `src/fhr/world/`, `src/fhr/evals/` and `scripts/` read customers' tastes,
  merchants' quality or real kitchen capacity. Policies and models get the catalog, the live context and the logs.
- **Training sees what serving sees.** Features come from `features.sql` offline and `online()` at serving time;
  `tests/test_features.py` keeps them equal. Serve-time values (distance, ETA, load) are logged with the impression.
- **Same future for every policy.** Policies are compared on forks of the same world with the same sessions and
  random numbers. A comparison that breaks this is not a comparison.
- **The gate does not read the truth.** It decides from the experiment a team could run and the serving benchmark.
- **Settings in config.** Every threshold and weight lives in `config/platform.toml`; the truth in `config/world*.toml`.
- **Reports are generated.** `reports/*.json`, `web/public/data.json` and the README results come from scripts.

## The held-out rule

`config/world_heldout.toml` is a city nobody tuned against. `scripts/heldout.py` runs it once, refuses to run if a
file pinned in `config/heldout.lock` changed, and refuses a second run.
