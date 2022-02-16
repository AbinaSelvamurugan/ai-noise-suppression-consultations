.PHONY: install api frontend test lint fixture train benchmark compose
install:
	python -m pip install -e '.[evaluation,quality]'
api:
	PYTHONPATH=src:. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
frontend:
	cd frontend && npm run dev
test:
	PYTHONPATH=src:. python -m pytest -q
lint:
	ruff check src backend scripts tests
fixture:
	PYTHONPATH=src:. python scripts/generate_fixture.py
	PYTHONPATH=src:. python scripts/prepare_dataset.py --root data/synthetic --output artifacts/manifest.jsonl
train:
	PYTHONPATH=src:. python scripts/train.py --root data/synthetic --manifest artifacts/manifest.jsonl --output artifacts/software-test-run --epochs 1
benchmark:
	PYTHONPATH=src:. python scripts/benchmark.py --root data/synthetic --manifest artifacts/manifest.jsonl --output artifacts/software-test-benchmark --max-examples 2
compose:
	docker compose up --build
