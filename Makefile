.PHONY: test lint spark evaluate experiments bench gate report web
test:
	pytest -q
lint:
	ruff check . && ruff format --check .
spark:
	pytest -q -m spark
evaluate:
	python scripts/evaluate.py
experiments:
	python scripts/experiments.py
bench:
	python scripts/serve_bench.py
gate:
	python scripts/gate.py
report:
	python scripts/report.py
web:
	cd web && npm ci && npm test && npm run build
