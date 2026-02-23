FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (stub package for layer caching)
COPY pyproject.toml README.md ./
RUN mkdir -p src/ecodiaos && touch src/ecodiaos/__init__.py
RUN pip install --upgrade pip && pip install .

# Copy real source and reinstall package (deps already cached)
COPY src/ src/
COPY config/ config/
RUN pip install --no-deps --force-reinstall .

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Create non-root user with home directory for HF cache
RUN groupadd -r ecodiaos && useradd -r -g ecodiaos -m ecodiaos
RUN cp -r /root/.cache /home/ecodiaos/.cache && chown -R ecodiaos:ecodiaos /home/ecodiaos/.cache
USER ecodiaos

EXPOSE 8000 8001 8002

CMD ["uvicorn", "ecodiaos.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop"]
