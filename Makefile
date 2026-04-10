.PHONY: install lint typecheck test check dev compose-up compose-down compose-logs

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

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f
