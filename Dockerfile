FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8080

# Cloud Run sets PORT; default matches Cloud Run’s convention.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
