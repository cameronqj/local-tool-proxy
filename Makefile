.PHONY: help install dev test lint demo run clean

help:
	@echo "Targets:"
	@echo "  install   Install the package"
	@echo "  dev       Install with dev extras (pytest, ruff, openai)"
	@echo "  test      Run the test suite"
	@echo "  lint      Run ruff"
	@echo "  demo      Run the no-Ollama repair demo"
	@echo "  run       Start the proxy against a local Ollama on :9000"
	@echo "  clean     Remove caches and build artifacts"

install:
	python3 -m pip install -e .

dev:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest

lint:
	python3 -m ruff check .

demo:
	python3 demo.py

run:
	python3 -m proxy.server \
		--host 127.0.0.1 \
		--port 9000 \
		--ollama-base http://localhost:11434/v1 \
		--compat-models gemma4:e4b-mlx,gemma4:e2b-mlx,gpt-oss:20b

clean:
	rm -rf .pytest_cache .ruff_cache .tmp_pycache build dist *.egg-info local_tool_proxy.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
