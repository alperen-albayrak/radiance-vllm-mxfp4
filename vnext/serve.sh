#!/usr/bin/env bash
# Serve an MXFP4 Qwen3.8-27B-class model with the radiance kernels on a current vLLM image
# (overlay in this directory; base and image in ./BASE). Two R9700s, TP=2. See README.md.
#
#   ./serve.sh                       DaisyRose-27B + DFlash2-FP8 drafter, port 8080
#   PORT=8090 NAME=radiance-test ./serve.sh      a second instance for tests (needs free GPUs)
#   PREPARE_ONLY=1 ./serve.sh        build kernels + apply overlay, no serve (no GPU needed)
#   DRY_RUN=1 ./serve.sh             print the container command, change nothing
#
# Environment (defaults in brackets):
#   MODELS [~/models]                directory mounted read-only at /models
#   MODEL  [DaisyRose-27B-MXFP4-mtpfp8]   target, a directory under MODELS
#   DRAFTER [Qwen3.8-27B-DFlash2-FP8]     DFlash drafter under MODELS; DRAFTER= serves without one
#   SPEC                             full --speculative-config JSON; overrides DRAFTER
#   SERVED_NAMES [DaisyRose-27B]     model name(s) clients ask for
#   NAME [radiance-vnext]  PORT [8080]  HOST [0.0.0.0]  GPUS [0,1]
#   MAXLEN [262144]  MAXSEQS [8]  CHUNK [8192]  GPU_UTIL [0.90]  TOOLS [1]
#   GEN ['{"temperature":0.7,"top_p":0.95,"top_k":20}']   default sampling for requests without their own
#   R4D_AR [1]                       radiance all-reduce between the cards (0 = RCCL only)
#   EXTRA                            more `vllm serve` arguments
#   ENVX                             a second env file, read after radiance-tp2.env (overrides it)
#   STATE [~/.cache/radiance-vnext]  built kernels and compile caches, one subdirectory per image
#   R4D_SRC [~/.cache/radiance-libr4d/b9e42ab-rx9]   libr4d source checkout
#   DETACH [1]                       0 = stay attached and print the log
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
die() { echo "serve.sh: $*" >&2; exit 1; }

IMAGE=${IMAGE:-$(awk '{print $NF; exit}' "$HERE/BASE")}
MODELS=${MODELS:-$HOME/models}
MODEL=${MODEL:-DaisyRose-27B-MXFP4-mtpfp8}
DRAFTER=${DRAFTER-Qwen3.8-27B-DFlash2-FP8}
SERVED_NAMES=${SERVED_NAMES:-DaisyRose-27B}
NAME=${NAME:-radiance-vnext}; PORT=${PORT:-8080}; HOST=${HOST:-0.0.0.0}; GPUS=${GPUS:-0,1}
MAXLEN=${MAXLEN:-262144}; MAXSEQS=${MAXSEQS:-8}; CHUNK=${CHUNK:-8192}; GPU_UTIL=${GPU_UTIL:-0.90}
GEN=${GEN:-'{"temperature":0.7,"top_p":0.95,"top_k":20}'}
STATE=${STATE:-$HOME/.cache/radiance-vnext}
R4D_SRC=${R4D_SRC:-$HOME/.cache/radiance-libr4d/b9e42ab-rx9}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen-fixed-v22.3.jinja}
TP=$(awk -F, '{print NF}' <<<"$GPUS")
[ "$TP" = 2 ] || die "this launcher and radiance-tp2.env are set up for two cards (GPUS=$GPUS)"

[ -d "$MODELS/$MODEL" ] || die "target not found: $MODELS/$MODEL"
if [ -z "${SPEC+x}" ]; then
  if [ -n "$DRAFTER" ]; then
    [ -d "$MODELS/$DRAFTER" ] || die "drafter not found: $MODELS/$DRAFTER (DRAFTER= serves without one)"
    # The drafter keeps vLLM's Triton attention: the R4D backend is built for the target's head shape.
    SPEC='{"method":"dflash","model":"/models/'$DRAFTER'","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN","draft_sample_method":"greedy"}'
  else
    SPEC=
  fi
fi
[ -d "$R4D_SRC" ] || die "libr4d source not found: $R4D_SRC"
[ -f "$REPO/$CHAT_TEMPLATE" ] || die "chat template not found: $REPO/$CHAT_TEMPLATE"

# Kernels are built against one image's Python and ROCm, so they live per image.
KEY=$(printf '%s' "$IMAGE" | sha256sum | cut -c1-12)
BUILD=$STATE/$KEY/build; CACHE=$STATE/$KEY/cache
HIP_IDS=$(seq -s, 0 $((TP - 1)))   # ROCR_VISIBLE_DEVICES filters by absolute id; HIP ids index into that list

CMD=(docker run --name "$NAME" --privileged --ipc host --network host --shm-size 16g
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined
  --device /dev/kfd --device /dev/dri
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)"
  --env-file "$HERE/radiance-tp2.env" ${ENVX:+--env-file "$ENVX"}
  -e ROCR_VISIBLE_DEVICES="$GPUS" -e HIP_VISIBLE_DEVICES="$HIP_IDS"
  -e RADIANCE_USE_R4D_AR="${R4D_AR:-1}"
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter
  -e PREPARE_ONLY="${PREPARE_ONLY:-0}" -e MODEL="$MODEL" -e SERVED_NAMES="$SERVED_NAMES"
  -e HOST="$HOST" -e PORT="$PORT" -e TP="$TP" -e MAXLEN="$MAXLEN" -e MAXSEQS="$MAXSEQS"
  -e CHUNK="$CHUNK" -e GPU_UTIL="$GPU_UTIL" -e SPEC="$SPEC" -e GEN="$GEN" -e TOOLS="${TOOLS:-1}"
  -e CHAT_TEMPLATE="$CHAT_TEMPLATE" -e EXTRA="${EXTRA:-}"
  -v "$MODELS:/models:ro" -v "$REPO:/patches:ro" -v "$HERE:/overlay:ro" -v "$R4D_SRC:/r4d-src:ro"
  -v "$BUILD:/build" -v "$CACHE:/cache"
  --entrypoint bash "$IMAGE" /overlay/entrypoint.sh)
[ "${DETACH:-1}" = 1 ] && CMD=("${CMD[@]:0:2}" -d "${CMD[@]:2}")

if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
if [ "${PREPARE_ONLY:-0}" != 1 ] && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  die "port $PORT is in use (stop the other server first, or pick another PORT)"
fi
mkdir -p "$BUILD" "$CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "[run] $IMAGE | $NAME | port $PORT | $MODEL as '$SERVED_NAMES' | $([ -n "$SPEC" ] && echo "speculative: ${DRAFTER:-custom}" || echo "no drafter")" >&2
"${CMD[@]}"
[ "${DETACH:-1}" = 1 ] && echo "[run] log: docker logs -f $NAME    stop: docker stop $NAME" >&2
