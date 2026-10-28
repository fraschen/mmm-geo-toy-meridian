# Reproducible environment from requirements.lock (Python 3.11, JAX backend).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock .
RUN python -m venv .venv && .venv/bin/pip install --no-cache-dir -r requirements.lock

COPY . .
# Keeps the default backend (JAX); MERIDIAN_BACKEND is deliberately not set.
# Default command: the production pipeline for the single model in config/model.toml.
CMD ["make", "all"]
