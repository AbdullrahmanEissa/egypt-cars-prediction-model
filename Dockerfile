FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
RUN python train.py && chown -R appuser:appuser /app

USER appuser

EXPOSE 6000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:6000/')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:6000", "--workers", "2", "--threads", "4", "--timeout", "60", "app:app"]
