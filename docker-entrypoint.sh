#!/usr/bin/env sh
# One image, several roles -- the sibling crawler's pattern. Unlike that project,
# this service owns the schema, so `migrate` runs before `serve` rather than being
# someone else's job.
#
#   serve        apply migrations, run the API, AND hold the daily crawl schedule
#                (01:00 CRAWL_TIMEZONE, in-process) -- the default, and all a
#                Railway deployment needs; see app/scheduler.py
#   ingest       one INCREMENTAL harvest cycle, then exit -- the batch alternative,
#                for a platform cron ('0 18 * * *' = daily at 01:00 WIB). Set
#                CRAWL_SCHEDULE_ENABLED=false on the API when you use this, or the
#                two schedules both wake up for the same slot.
#   ingest-full  one full harvest cycle (first deploy, or corpus rebuild), then exit
#   classify     (re)run topic classification, then exit
#   migrate      apply migrations and exit
#   reindex      rebuild the BM25 artifact and exit
#   worker       crawl daily at 01:00 Asia/Jakarta in a loop, with no HTTP -- for
#                docker-compose and anything that wants the cadence in its own
#                process. `serve` already schedules; this is the standalone twin.
#   <other>      executed verbatim, so `docker run ... sh` still works
set -e
role="${1:-serve}"; shift 2>/dev/null || true
case "$role" in
  serve)
    python -m scripts.init_db
    # --workers 1 deliberately: each uvicorn worker is a separate process loading its
    # own copy of torch, the e5 model and the BM25 artifact (~700 MB RSS each). Scale
    # by replicas if ever needed, not by workers. --timeout-keep-alive exceeds
    # CHAT_TIMEOUT_SECONDS so uvicorn never closes a stream the app still writes to.
    exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}" \
         --workers 1 --timeout-keep-alive 75 --no-access-log "$@"
    ;;
  ingest)      python -m scripts.init_db; exec python -m scripts.run_ingest --incremental "$@" ;;
  ingest-full) python -m scripts.init_db; exec python -m scripts.run_ingest "$@" ;;
  classify)    exec python -m scripts.classify_topics "$@" ;;
  migrate)     exec python -m scripts.init_db "$@" ;;
  reindex)     exec python -m scripts.rebuild_index "$@" ;;
  worker)      python -m scripts.init_db; exec python -m scripts.run_worker "$@" ;;
  *)           exec "$role" "$@" ;;
esac
