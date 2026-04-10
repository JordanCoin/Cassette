.PHONY: install lint typecheck test check dev

install:
	uv venv
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .

typecheck:
	uv run mypy .

test:
	uv run pytest -q

check: lint typecheck test

dev:
	uv run uvicorn services.gateway.app:app --reload --port 8000
