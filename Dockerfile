# --- builder ---------------------------------------------------------------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# The --extra-index-url for the CPU torch wheel lives in requirements.txt itself, so
# `pip wheel` here and `pip install` below resolve torch identically. The +cpu local
# version only exists on that index -- the default PyPI wheel bundles ~2.5 GB of CUDA.
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# --- runtime ---------------------------------------------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app HF_HOME=/app/.hf \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels

# Bake the embedding model into the image. Downloading 470 MB on first request makes
# cold start a coin flip on Hugging Face's availability, and Railway's health check
# would fail the deploy while it downloaded.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "import os;from sentence_transformers import SentenceTransformer;\
SentenceTransformer(os.environ['EMBEDDING_MODEL'])"
# After baking, never reach for the network for model files again: a Hugging Face
# outage becomes a non-event instead of a cold-start failure.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY . .
RUN chmod +x docker-entrypoint.sh \
 && useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser
EXPOSE 8080

# /health answers "is the process up", not "is the corpus healthy". A failed ingest
# must not fail the probe and get the API restarted mid-conversation.
# start-period is generous because the lifespan loads torch + the e5 model + the BM25
# artifact -- 10-20s on a shared CPU, and a short start period causes a restart loop.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import os,sys,urllib.request;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8080')+'/health',timeout=8).status==200 else 1)"
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["serve"]
