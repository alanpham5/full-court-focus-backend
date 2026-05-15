#!/usr/bin/env bash
set -euo pipefail

# Catch up if the container was down across a scheduled run.
/usr/local/bin/run-scrape.sh || true

exec cron -f
