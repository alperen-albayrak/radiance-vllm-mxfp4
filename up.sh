#!/usr/bin/env bash
# up.sh -- start the server on one or two R9700s, with the settings in local.env. SERVING.md
# explains it.
#
#   ./up.sh            start (two cards: the current-vLLM serve in vnext/; one card: the 0.9.3
#                      image's single-GPU profile), then wait until it answers
#   ./up.sh --print    print the command it would run, start nothing
#
# Stop with ./down.sh. Everything else is local.env (copy local.env.example).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
die() { echo "up.sh: $*" >&2; exit 1; }
[ -f "$HERE/local.env" ] && { set -a; . "$HERE/local.env"; set +a; }

MODELS=${MODELS:-$HOME/models}
MODEL=${MODEL:-qwen3.8}
case "$MODEL" in
  qwen3.8)   DIR=Qwen3.8-27B-MXFP4-mtpfp8; NAMES_DEFAULT="Qwen3.8"; TEMP=0.7 ;;
  # RadixArk recommends temperature 1.0 for DaisyRose with thinking on
  daisyrose) DIR=DaisyRose-27B-MXFP4-mtpfp8; NAMES_DEFAULT="DaisyRose-27B"; TEMP=1.0 ;;
  *)         DIR=$MODEL; NAMES_DEFAULT=$MODEL; TEMP=0.7 ;;   # any checkpoint directory under MODELS
esac
GEN=${GEN:-'{"temperature":'$TEMP',"top_p":0.95,"top_k":20}'}
SERVED_NAMES=${SERVED_NAMES:-$NAMES_DEFAULT}
PORT=${PORT:-8080}
NAME=radiance
[ -f "$MODELS/$DIR/config.json" ] || die "no checkpoint in $MODELS/$DIR: run MODEL=$MODEL ./setup-mxfp4.sh first"

. "$HERE/gpu-detect.sh"
CARDS=${CARDS:-$(( RAD_GPU_COUNT >= 2 ? 2 : RAD_GPU_COUNT ))}
[ "$CARDS" -ge 1 ] || die "no usable AMD GPU found (./gpu-detect.sh shows what is there)"
GPUS=${GPUS:-$(cut -d, -f1-"$CARDS" <<<"$RAD_GPU_INDICES")}

PRINT=; [ "${1:-}" = --print ] && PRINT=1
if [ -z "$PRINT" ] && docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  die "it is already running (./down.sh stops it)"
fi

export MODELS SERVED_NAMES PORT NAME GPUS GEN
case "$CARDS" in
  2)
    echo "[up] two cards ($GPUS): $DIR as '$SERVED_NAMES', current vLLM (vnext/)"
    # 4 parallel requests: as fast as 8 at 1-3 sessions, and ~1 GiB more VRAM free per card
    MAXSEQS=${MAXSEQS:-4} MODEL=$DIR DRY_RUN=$PRINT "$HERE/vnext/serve.sh" ;;
  1)
    echo "[up] one card ($GPUS): $DIR as '$SERVED_NAMES', image 0.9.3, single-GPU profile"
    # embed-host moves the 2.4 GiB input embedding to host memory: room for ~60k more KV tokens
    SNAP=$MODELS/$DIR RUNTIME=docker DETACH=1 RADIANCE_EMBED_HOST=${RADIANCE_EMBED_HOST:-1} \
      DRY_RUN=$PRINT "$HERE/serve-tp1.sh" ;;
  *) die "CARDS must be 1 or 2 (got $CARDS)" ;;
esac
[ -n "$PRINT" ] && exit 0

echo "[up] waiting for http://127.0.0.1:$PORT/v1/models (the first start compiles kernels: several minutes)"
for _ in $(seq 1 180); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    echo "[up] ready: http://<this host>:$PORT/v1, model '${SERVED_NAMES%% *}'"; exit 0
  fi
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { docker logs --tail 30 "$NAME" 2>&1; die "the server stopped while starting (log above)"; }
  sleep 5
done
die "no answer after 15 minutes: docker logs -f $NAME"
