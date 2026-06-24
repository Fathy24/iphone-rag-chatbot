# syntax=docker/dockerfile:1

# -----------------------------------------------------------------------------
# Single-container image for the iPhone User Guide RAG chatbot.
# Build:  docker build -t chatbot:1.0 .
# Run:    docker run -p 8000:8000 --env-file .env chatbot:1.0
# (Override the port with PORT in your .env; map the same port with -p.)
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Predictable, log-friendly Python in containers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install dependencies first to maximise Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY app ./app
# Front-end assets: custom JSX elements (Dashboard / ExecutionTrace / AnswerTools
# / TestCases) plus custom.css / custom.js. Without these the transparent UI
# won't render.
COPY public ./public
COPY chainlit.md ./chainlit.md
COPY .chainlit ./.chainlit
COPY docker-entrypoint.sh ./docker-entrypoint.sh
# No retrieval index files are bundled: this image is cloud-only. The Qdrant
# collection is the single source of truth — at startup the app streams the
# chunk corpus (payloads + vectors) from Qdrant and rebuilds the BM25 sparse
# index, the section centroids (coarse stage) and the parent-document expander
# in memory. So HYBRID + HIERARCHICAL retrieval work with nothing local. (The
# local FAISS index, the BM25/section JSON dumps and the source PDF are used
# only for offline local development — see the README.)

# Run as a non-root user.
RUN sed -i 's/\r$//' docker-entrypoint.sh \
    && chmod +x docker-entrypoint.sh \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Documented default; the actual published port follows $PORT at runtime.
EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
