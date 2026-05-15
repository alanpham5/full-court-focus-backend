#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
cd "$APP_DIR"

echo "[$(date -Iseconds)] Checking scrape schedule..."
python scripts/check_and_scrape.py
echo "[$(date -Iseconds)] Check finished."

if [[ "${GIT_PUSH:-false}" != "true" ]]; then
  exit 0
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "GIT_PUSH=true but GITHUB_TOKEN is not set; skipping git push."
  exit 0
fi

REPO_ROOT="${REPO_ROOT:-/workspace}"
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:?Set GITHUB_REPOSITORY, e.g. owner/full-court-focus-backend}"

cd "$REPO_ROOT"
git config user.name "${GIT_USER_NAME:-nba-scraper}"
git config user.email "${GIT_USER_EMAIL:-nba-scraper@users.noreply.github.com}"

git add app/data/static/
if git diff --staged --quiet; then
  echo "No data changes to commit."
  exit 0
fi

git commit -m "chore: refresh NBA data $(date +%Y-%m-%d)"
git push "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" HEAD

echo "[$(date -Iseconds)] Pushed updated data to GitHub."
