FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_DIR=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends cron git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY docker/scraper/run-scrape.sh /usr/local/bin/run-scrape.sh
COPY docker/scraper/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/scraper/cron/scrape /etc/cron.d/nba-scrape

RUN chmod +x /usr/local/bin/run-scrape.sh /usr/local/bin/entrypoint.sh \
    && chmod 0644 /etc/cron.d/nba-scrape \
    && touch /var/log/scrape.log

# Daily 6:05 UTC check; also runs once at container start (see entrypoint.sh).
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
