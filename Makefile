.PHONY: install test lint format coverage build demo clean

install:
	python3 -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

coverage:
	pytest --cov=robot_dataset_tools --cov-report=term-missing

build:
	python3 -m build
	python3 -m twine check dist/*

demo:
	python3 scripts/create_demo_data.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov reports outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

