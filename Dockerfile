FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY memory_vault ./memory_vault

RUN pip install --upgrade pip \
    && pip install ".[all]" \
    && mkdir -p /app/data

ENV ML_HOST=0.0.0.0 \
    ML_PORT=8000 \
    ML_CHROMA_PATH=/app/data/chroma \
    ML_SQLITE_PATH=/app/data/memory.db

EXPOSE 8000
VOLUME ["/app/data"]

CMD ["uvicorn", "memory_vault.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
