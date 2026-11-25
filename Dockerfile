FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py main.py ./

RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

EXPOSE 8000

# Render injects PORT (default 10000) and expects the app to bind to it;
# shell form (not exec form) is required for ${PORT} to expand.
# Local docker run without -e PORT falls back to 8000.
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}