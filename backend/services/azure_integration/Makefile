# Makefile for Azure Integration Service

.PHONY: help install dev test lint format clean build run docker-build docker-run docker-stop

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean up build artifacts"
	@echo "  build       - Build the application"
	@echo "  run         - Run the application"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run with Docker Compose"
	@echo "  docker-stop - Stop Docker containers"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov black flake8 isort

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-coverage:
	pytest --cov=. --cov-report=html --cov-report=term tests/

# Run linting
lint:
	flake8 --max-line-length=100 --ignore=E203,W503 .
	black --check .
	isort --check-only .

# Format code
format:
	black .
	isort .

# Clean up build artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

# Build the application
build:
	python -m build

# Run the application
run:
	python main.py

# Run with development server
dev-run:
	uvicorn main:app --reload --host 0.0.0.0 --port 8001

# Docker commands
docker-build:
	docker build -t azure-integration-service .

docker-run:
	docker-compose up --build

docker-stop:
	docker-compose down

docker-clean:
	docker-compose down -v
	docker system prune -f

# Health check
health:
	curl -f http://localhost:8001/health || echo "Service not healthy"

# Check if service is ready
ready:
	curl -f http://localhost:8001/ready || echo "Service not ready"

# View metrics
metrics:
	curl http://localhost:8001/metrics

# Setup development environment
setup-dev:
	python -m venv venv
	@echo "Activate virtual environment with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
	@echo "Then run: make dev"

# Generate API documentation
docs:
	@echo "API documentation available at: http://localhost:8001/docs"
	@echo "ReDoc documentation available at: http://localhost:8001/redoc"