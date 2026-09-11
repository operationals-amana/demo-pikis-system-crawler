#!/usr/bin/env bash
# Local runner for the PIKIS backend (demo-pikis-system-crawler).
#
# The local twin of docker-entrypoint.sh: same role words, but sourcing the pyenv
# virtualenv instead of living in a container. Anything after the role is passed
# through, so `./run.sh serve --reload` and `./run.sh crawl --source ije` both work.
#
#   ./run.sh                 serve the API on :8000 (the default)
#   ./run.sh serve --reload  serve with auto-reload (pair with EMBEDDING_LAZY=true)
#   ./run.sh ingest          one INCREMENTAL cycle: Scrapy crawl -> ... -> index
#   ./run.sh ingest-full     full harvest (first run, or corpus rebuild)
#   ./run.sh crawl           Scrapy spiders only, no downstream stages
#   ./run.sh classify        (re)run topic classification
#   ./run.sh worker          crawl daily at 01:00 Asia/Jakarta, in a loop
#   ./run.sh eval            retrieval quality gate (fused recall@5 must be >= 0.80)
#   ./run.sh migrate         create the database if missing + apply migrations
#   ./run.sh admin -- --email you@x.id --password '...'   create/reset an admin
set -euo pipefail

# Always run from the repo root so `python -m scripts.*` and scrapy.cfg resolve,
# no matter where the script was invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV="${PIKIS_VENV:-$HOME/.pyenv/versions/pikis}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "error: virtualenv not found at $VENV" >&2
  echo "       create it, or point PIKIS_VENV at an existing one:" >&2
  echo "       pyenv virtualenv 3.11.11 pikis && $VENV/bin/pip install --no-cache-dir -r requirements.txt" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=.

if [[ ! -f .env ]]; then
  echo "warning: no .env file -- copying .env.example; set JWT_SECRET and ANTHROPIC_API_KEY" >&2
  cp .env.example .env
fi
# The two settings whose absence fails at runtime rather than at startup -- say so
# now instead of during the first login or the first chat.
grep -q '^JWT_SECRET=..*' .env        || echo "warning: JWT_SECRET is empty in .env -- auth will refuse to mint tokens" >&2
grep -q '^ANTHROPIC_API_KEY=..*' .env || echo "warning: ANTHROPIC_API_KEY is empty in .env -- chat and classification degrade" >&2

role="${1:-serve}"
shift 2>/dev/null || true
# Allow `./run.sh admin -- --email ...` so role flags never collide with ours.
[[ "${1:-}" == "--" ]] && shift

case "$role" in
  serve)
    python -m scripts.init_db
    # --timeout-keep-alive must exceed CHAT_TIMEOUT_SECONDS (default 90 is fine at
    # 75s keep-alive because SSE heartbeats keep the connection non-idle) so uvicorn
    # never closes a stream the app still intends to write to.
    exec uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8001}" \
         --timeout-keep-alive 75 "$@"
    ;;
  ingest)      python -m scripts.init_db; exec python -m scripts.run_ingest --incremental "$@" ;;
  ingest-full) python -m scripts.init_db; exec python -m scripts.run_ingest "$@" ;;
  crawl)       exec python -m scripts.run_crawl "$@" ;;
  classify)    exec python -m scripts.classify_topics "$@" ;;
  worker)      python -m scripts.init_db; exec python -m scripts.run_worker "$@" ;;
  eval)        exec python -m scripts.eval_retrieval "$@" ;;
  migrate)     exec python -m scripts.init_db "$@" ;;
  reindex)     exec python -m scripts.rebuild_index "$@" ;;
  admin)       exec python -m scripts.create_admin "$@" ;;
  help|-h|--help)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    echo "error: unknown role '$role' -- try: ./run.sh help" >&2
    exit 1
    ;;
esac
