# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit

ARG PYTHON_VERSION=3.11.7
ARG PORT=8000
################################
# PYTHON-BASE
# Sets up all our shared environment variables
################################
FROM python:${PYTHON_VERSION}-slim as python-base

    # Python
ENV PYTHONUNBUFFERED=1 \
    # pip
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # Poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.7.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # never create virtual environment automaticly, only use env prepared by us
    POETRY_VIRTUALENVS_CREATE=false \
    # this is where our requirements + virtual environment will live
    VIRTUAL_ENV="/venv"

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENV/bin:$PATH"

# prepare virtual env
RUN python -m venv $VIRTUAL_ENV

# working directory and Python path
WORKDIR /app
ENV PYTHONPATH="/app:$PYTHONPATH"

################################
# BUILDER-BASE
# Used to build deps + create our virtual environment
################################
FROM python-base as builder-base
RUN apt-get update && \
    apt-get install -y \
    apt-transport-https \
    gnupg \
    ca-certificates \
    build-essential \
    git \
    vim \
    curl

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
# The --mount will mount the buildx cache directory to where
# Poetry and Pip store their cache so that they can re-use it
RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python -

# used to init dependencies
WORKDIR /app
COPY pyproject.toml ./
RUN poetry lock

# install runtime deps to VIRTUAL_ENV
RUN --mount=type=cache,target=/root/.cache \
    poetry install --no-root --all-extras --only main


################################
# DEVELOPMENT
# Image used during development / testing
################################
FROM builder-base as development

WORKDIR /app
COPY --from=builder-base /app/pyproject.toml pyproject.toml
COPY --from=builder-base /app/poetry.lock poetry.lock


# quicker install as runtime deps are already installed
RUN --mount=type=cache,target=/root/.cache \
    poetry install --no-root --all-extras --with dev

COPY . .
RUN poetry install --all-extras --only-root

ARG PORT
EXPOSE $PORT

CMD ["bash"]


################################
# PRODUCTION
# Final image used for runtime
################################
FROM python-base as production

ENV WORKER_COUNT=1

LABEL org.opencontainers.image.source="https://github.com/pinecone-io/canopy"
LABEL org.opencontainers.image.description="Image containing the canopy server."
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates && \
    apt-get clean

# copy in our built poetry + venv
COPY --from=builder-base $POETRY_HOME $POETRY_HOME
COPY --from=builder-base $VIRTUAL_ENV $VIRTUAL_ENV

WORKDIR /app

COPY --from=builder-base /app/pyproject.toml pyproject.toml
COPY --from=builder-base /app/poetry.lock poetry.lock

COPY src/ src/
COPY config/ config/
RUN touch README.md
RUN poetry install --all-extras --only-root

ARG PORT
EXPOSE $PORT
ENV PORT $PORT
CMD ["sh", "-c", "gunicorn canopy_server.app:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers $WORKER_COUNT"]


