#!/usr/bin/env bash
# Run ttl_index.py until it completes. It is incremental, so a run stopped by a
# timeout loses at most one batch and the next run carries on from there.
#
#     setsid nohup src/ttl_index_until_done.sh >/dev/null 2>&1 &
#
# Writes logs/ttl_index.log. Stops after MAX_STALLED runs in a row that do not
# reduce the chunks left to embed, because retrying will not fix that.
cd "$(dirname "$0")/.." || exit 1
LOG=logs/ttl_index.log
MAX_STALLED=5
WAIT=60          # seconds between runs
RUN_LIMIT=6h     # a run that hangs is killed after this

left() {         # chunks the latest run still had to embed
    grep -E '^To embed' "$LOG" | tail -1 | grep -oE ':[[:space:]]+[0-9]+' | grep -oE '[0-9]+'
}

: > "$LOG"
run=0 stalled=0 best=""
while :; do
    run=$((run + 1))
    echo "=== run $run, $(date '+%F %T')" >> "$LOG"
    PYTHONPATH=src timeout -k 60 "$RUN_LIMIT" .venv/bin/python -u src/ttl_index.py >> "$LOG" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        echo "=== DONE after $run run(s), $(date '+%F %T')" >> "$LOG"
        exit 0
    fi
    now=$(left)
    echo "=== run $run stopped with exit code $code; $now chunks were left at its start" >> "$LOG"
    if [ -n "$now" ] && { [ -z "$best" ] || [ "$now" -lt "$best" ]; }; then
        best=$now stalled=0
    else
        stalled=$((stalled + 1))
    fi
    if [ "$stalled" -ge "$MAX_STALLED" ]; then
        echo "=== GAVE UP: $stalled runs without progress, $(date '+%F %T')" >> "$LOG"
        exit 1
    fi
    sleep "$WAIT"
done
