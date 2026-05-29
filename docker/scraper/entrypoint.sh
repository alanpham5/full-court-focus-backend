#!/usr/bin/env bash
set -euo pipefail

/usr/local/bin/run-scrape.sh || true

exec cron -f
