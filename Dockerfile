FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ATLAS_DATA_DIR=/data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
RUN python -m pip install --no-cache-dir '.[api,gemini,langgraph,observability]'

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn atlas.api.app:app --host 0.0.0.0 --port ${ATLAS_PORT:-8000}"]
