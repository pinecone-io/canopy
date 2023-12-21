IMAGE_NAME = canopy
DOCKERFILE_DIR = .
COMMON_BUILD_ARGS = --progress plain
EXTRA_BUILD_ARGS =

.PHONY: build build-dev run run-dev help

build:
	@echo "Building Docker image..."
	docker build $(COMMON_BUILD_ARGS) $(EXTRA_BUILD_ARGS) -t $(IMAGE_NAME) $(DOCKERFILE_DIR)
	@echo "Docker build complete."

build-dev:
	@echo "Building Docker image for development..."
	docker build $(COMMON_BUILD_ARGS) $(EXTRA_BUILD_ARGS) -t $(IMAGE_NAME)/dev --target=development $(DOCKERFILE_DIR)
	@echo "Development Docker build complete."

run:
	docker run --env-file .env -p 8000:8000 $(IMAGE_NAME)

run-dev:
	docker run --env-file .env -p 8000:8000 $(IMAGE_NAME)/dev

help:
	@echo "Available targets:"
	@echo "  make build       - Build the Docker image."
	@echo "  make build-dev   - Build the Docker image for development."
	@echo "  make run         - Run the Docker image."
	@echo "  make run-dev     - Run the Docker image for development."
