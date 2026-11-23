#!/usr/bin/env bash
# onecard-gate.sh -- the gate for serving ONE card on this base: run the same checks and
# measurements on the 0.9.3 single-GPU profile (serve-tp1.sh) and on vnext at TP=1 (serve.sh with
# one card), one after the other on the same card, and keep everything in one directory.
#
#   vnext/tests/onecard-gate.sh                 card 0, DaisyRose, results in ./onecard-gate-<date>
#   CARD=1 ENGINES=vnext vnext/tests/onecard-gate.sh       one engine only
#
# Needs the card free (nothing else on it) and ~3 hours for both engines. Environment:
#   CARD [0]  MODELS [~/models]  MODEL [DaisyRose-27B-MXFP4-mtpfp8]  NAMES [DaisyRose-27B]
#   PORT [8181]  ENGINES ["093 vnext"]  SOAK_MIN [20]  OUT [./onecard-gate-<date>]
#   SIZES [1000,8000,32000,128000,200000]  SESSIONS [1,2,3]   bench_ctx.py grid
# What decides "vnext replaces 0.9.3 on one card": every functional and soak check passes, the
# KV pool is at least 0.9.3's, decode per session within ~3% and prefill within ~5% (the
# two-card port's own bar, RESULTS.md). The KV size it logs is what KV_MEM should then pin.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
CARD=${CARD:-0}; MODELS=${MODELS:-$HOME/models}; MODEL=${MODEL:-DaisyRose-27B-MXFP4-mtpfp8}
NAMES=${NAMES:-DaisyRose-27B}; PORT=${PORT:-8181}; ENGINES=${ENGINES:-"093 vnext"}
SOAK_MIN=${SOAK_MIN:-20}; OUT=${OUT:-$PWD/onecard-gate-$(date +%Y%m%d-%H%M)}
export SIZES=${SIZES:-1000,8000,32000,128000,200000} SESSIONS=${SESSIONS:-1,2,3}
URL=http://127.0.0.1:$PORT; NAME=radiance-gate
mkdir -p "$OUT"; SUMMARY=$OUT/summary.txt
say() { echo "$*" | tee -a "$SUMMARY"; }

[ -f "$MODELS/$MODEL/config.json" ] || { echo "no checkpoint at $MODELS/$MODEL" >&2; exit 1; }
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT is in use" >&2; exit 1; fi

start() {   # $1 = engine
  docker rm -f "$NAME" >/dev/null 2>&1
  case "$1" in
    093)   SNAP=$MODELS/$MODEL MODELS=$MODELS GPUS=$CARD PORT=$PORT NAME=$NAME RUNTIME=docker DETACH=1 \
             SERVED_NAMES=$NAMES RADIANCE_EMBED_HOST=1 "$REPO/serve-tp1.sh" ;;
    vnext) MODELS=$MODELS MODEL=$MODEL GPUS=$CARD PORT=$PORT NAME=$NAME SERVED_NAMES=$NAMES DETACH=1 \
             "$REPO/vnext/serve.sh" ;;
  esac
  for _ in $(seq 1 240); do
    curl -sf -m 3 "$URL/v1/models" >/dev/null && return 0
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || return 1
    sleep 5
  done
  return 1
}

model=${NAMES%% *}
for e in $ENGINES; do
  d=$OUT/$e; mkdir -p "$d"
  say "== $e on card $CARD ($(date +%H:%M))"
  if ! start "$e" > "$d/start.log" 2>&1; then
    docker logs "$NAME" > "$d/server.log" 2>&1; say "   did not start: $d/server.log"; docker rm -f "$NAME" >/dev/null 2>&1; continue
  fi
  docker logs "$NAME" 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|embed-host:|single-GPU profile|KV cache" | tee "$d/kv.txt" | sed 's/^/   /' | tee -a "$SUMMARY"
  python3 "$HERE/functest.py" "$URL" "$model" > "$d/functest.txt" 2>&1
  say "   functest: $(grep -c '^PASS' "$d/functest.txt") pass, $(grep -c '^FAIL' "$d/functest.txt") fail"
  for t in 1 0; do
    python3 "$HERE/cachecorrect2.py" "$URL" "$model" 30000 1 $t 0.7 >> "$d/cachecorrect.txt" 2>&1
  done
  say "   answer after a cache hit (thinking on, off): $(grep -o 'correct=[A-Za-z]*' "$d/cachecorrect.txt" | tr '\n' ' ')"
  python3 "$HERE/soak.py" "$URL" "$model" "$SOAK_MIN" 4 1 > "$d/soak.txt" 2>&1
  say "   soak ${SOAK_MIN} min: $(grep '^RESULT' "$d/soak.txt")"
  python3 "$HERE/specdec_compare.py" "$e" "$model" "$d/specdec.json" "$URL" > "$d/specdec.txt" 2>&1
  say "   one stream: $(tail -n 2 "$d/specdec.txt" | tr '\n' ' ' | cut -c1-200)"
  python3 "$HERE/bench_ctx.py" "$model" "$d/bench_ctx.json" "$URL" > "$d/bench_ctx.txt" 2>&1
  say "   context grid: $d/bench_ctx.txt"
  docker logs "$NAME" > "$d/server.log" 2>&1
  docker stop -t 30 "$NAME" >/dev/null
done
say "done: $OUT"
