.PHONY: install test demo calibrate run benchmark lint clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

demo:
	python main.py demo

calibrate:
	python main.py calibrate --board synthetic

calibrate-real:
	python main.py calibrate --board openbci_cyton

run:
	python main.py run

benchmark:
	python main.py benchmark

lint:
	ruff check src/ tests/ main.py
	ruff format --check src/ tests/ main.py

format:
	ruff format src/ tests/ main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache *.egg-info dist build
