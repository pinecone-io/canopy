TEST_WORKER_COUNT = 8

REPOSITORY = ghcr.io/pinecone-io/canopy

IMAGE_TAG = $(shell poetry version -s)
CONTAINER_PORT = 8000
CONTAINER_ENV_FILE = .env
CONTAINER_BUILD_DIR = .
CONTAINER_BUILD_PLATFORM = linux/amd64
CONTAINER_COMMON_BUILD_ARGS = --progress plain --platform $(CONTAINER_BUILD_PLATFORM) --build-arg PORT=$(CONTAINER_PORT)
CONTAINER_EXTRA_BUILD_ARGS =

# Only add the env file if it exists
CONTAINER_COMMON_RUN_ARGS = --platform linux/amd64 -p $(CONTAINER_PORT):$(CONTAINER_PORT) $(shell [ -e "$(CONTAINER_ENV_FILE)" ] && echo "--env-file $(CONTAINER_ENV_FILE)")
CONTAINER_EXTRA_RUN_ARGS =

.PHONY: lint static install install-extras test test-unit test-system test-e2e install docker-build docker-build-dev docker-run docker-run-dev help

lint:
	poetry run flake8 .

static:
	poetry run mypy src

install:
	poetry install

install-extras:
	poetry install --with dev --extras "$(CANOPY_EXTRAS)"

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
	docker build $(CONTAINER_COMMON_BUILD_ARGS) $(CONTAINER_EXTRA_BUILD_ARGS) -t $(REPOSITORY):$(IMAGE_TAG) $(CONTAINER_BUILD_DIR)
	@echo "Docker build complete."

docker-build-dev:
	@echo "Building Docker image for development..."
	docker build $(CONTAINER_COMMON_BUILD_ARGS) $(CONTAINER_EXTRA_BUILD_ARGS) -t $(REPOSITORY)-dev:$(IMAGE_TAG) --target=development $(CONTAINER_BUILD_DIR)
	@echo "Development Docker build complete."

docker-run:
	docker run $(CONTAINER_COMMON_RUN_ARGS) $(CONTAINER_EXTRA_RUN_ARGS) $(REPOSITORY):$(IMAGE_TAG)

docker-run-dev:
	docker run $(CONTAINER_COMMON_RUN_ARGS) $(CONTAINER_EXTRA_RUN_ARGS) -it $(REPOSITORY)-dev:$(IMAGE_TAG)

help:
	@echo "Available targets:"
	@echo ""
	@echo " -- DEV -- "
	@echo "  make install            - Install all the dependencies."
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
