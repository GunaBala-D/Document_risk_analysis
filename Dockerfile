# Backend image: FastAPI API, MLflow server and training jobs share this image so library
# versions (scikit-learn, LightGBM, skops, MLflow) are identical where models are trained,
# registered and loaded.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8 \
    MLFLOW_DISABLE_AGENT_HINT=1

# tesseract: OCR fallback for scanned contracts; libgomp1: OpenMP runtime for LightGBM/XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml alembic.ini ./
COPY configs ./configs
COPY src ./src
COPY api ./api
COPY database ./database
COPY training ./training
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p data/uploads data/processed data/synthetic artifacts \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["sh", "scripts/docker/api_entrypoint.sh"]
