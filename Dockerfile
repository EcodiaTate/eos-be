FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies (includes curl + git for elan/Lean 4 toolchain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 4A: Install Lean 4 toolchain via elan ──────────────────────────
# elan is the Lean version manager (like rustup for Rust).
# Installs to /root/.elan; the lean/lake binaries are symlinked into PATH.
# Set ELAN_DEFAULT_TOOLCHAIN to pin the Lean version for reproducibility.
ENV ELAN_HOME="/opt/elan" \
    PATH="/opt/elan/bin:${PATH}"
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- \
    --default-toolchain leanprover/lean4:v4.14.0 \
    --no-modify-path \
    -y \
    && lean --version \
    && lake --version

# Install Python dependencies (stub package for layer caching)
COPY pyproject.toml README.md ./
RUN mkdir -p ecodiaos && touch ecodiaos/__init__.py
RUN pip install --upgrade pip && pip install .

# Copy real source and reinstall package (deps already cached)
COPY ecodiaos/ ecodiaos/
COPY config/ config/
RUN pip install --no-deps --force-reinstall .

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Create non-root user with home directory for HF cache and elan access
RUN groupadd -r ecodiaos && useradd -r -g ecodiaos -m ecodiaos
RUN cp -r /root/.cache /home/ecodiaos/.cache && chown -R ecodiaos:ecodiaos /home/ecodiaos/.cache
# Grant non-root user read access to Lean toolchain
RUN chmod -R a+rX /opt/elan
USER ecodiaos

EXPOSE 8000 8001 8002

CMD ["uvicorn", "ecodiaos.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop"]
