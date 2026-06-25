FROM python:3.11-slim-bookworm

# Note: PYTHONDONTWRITEBYTECODE is intentionally NOT set — we want pip and the
# compileall step below to bake .pyc files into the image so cold starts don't
# pay to recompile every source file on first import.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY app/requirements.txt .
RUN pip install -r requirements.txt

COPY app/ .

# Pre-compile bytecode for the app and all installed packages so first import
# on a cold Cloud Run instance is faster.
RUN python -m compileall -q /app /usr/local/lib/python3.11/site-packages || true

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
