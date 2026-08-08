FROM python:3.11-slim

WORKDIR /app
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_DIR=/model/intent_classifier \
    INTENT_DEVICE=auto

COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.7.0 --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt

COPY runtime/app.py ./app.py
COPY model /model

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /model
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
