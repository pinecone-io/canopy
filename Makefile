IMAGE_NAME = canopy
DOCKERFILE_DIR = .
COMMON_BUILD_ARGS = --progress plain
EXTRA_BUILD_ARGS =
IMAGE_VERSION = $(shell poetry version -s)
PORT = 8000
ENV_FILE = .env
TEST_WORKER_COUNT = 8

.PHONY: lint static test test-unit test-system test-e2e docker-build docker-build-dev docker-run docker-run-dev help

lint:
	poetry run flake8 .

static:
	poetry run mypy src

test:
	poetry run pytest -n $(TEST_WORKER_COUNT) --dist loadscope

test-unit:
	poetry run pytest -n $(TEST_WORKER_COUNT) --dist loadscope tests/unit

test-system:
	poetry run pytest -n $(TEST_WORKER_COUNT) --dist loadscope tests/system

test-e2e:
	poetry run pytest -n $(TEST_WORKER_COUNT) --dist loadscope tests/e2e

docker-build:
	@echo "Building Docker image..."
	docker build $(COMMON_BUILD_ARGS) $(EXTRA_BUILD_ARGS) -t $(IMAGE_NAME):$(IMAGE_VERSION) $(DOCKERFILE_DIR)
	@echo "Docker build complete."

docker-build-dev:
	@echo "Building Docker image for development..."
	docker build $(COMMON_BUILD_ARGS) $(EXTRA_BUILD_ARGS) -t $(IMAGE_NAME)-dev:$(IMAGE_VERSION) --target=development $(DOCKERFILE_DIR)
	@echo "Development Docker build complete."

docker-run:
	docker run --env-file $(ENV_FILE) -p $(PORT):$(PORT) $(IMAGE_NAME):$(IMAGE_VERSION)

docker-run-dev:
	docker run -it --env-file $(ENV_FILE) -p $(PORT):$(PORT) $(IMAGE_NAME)-dev:$(IMAGE_VERSION)


help:
	@echo "Available targets:"
	@echo ""
	@echo " -- DEV -- "
	@echo "  make lint               - Lint the code."
	@echo "  make static             - Run static type checks."
	@echo "  make test               - Test the code."
	@echo "  make test-unit          - Run unit tests."
	@echo "  make test-system        - Run system tests."
	@echo "  make test-e2e           - Run e2e tests."
	@echo ""
	@echo " -- DOCKER -- "
	@echo "  make docker-build       - Build the Docker image."
	@echo "  make docker-build-dev   - Build the Docker image for development."
	@echo "  make docker-run         - Run the Docker image."
	@echo "  make docker-run-dev     - Run the Docker image for development."
