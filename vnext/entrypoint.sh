#!/usr/bin/env bash
# Runs inside the vLLM image (started by serve.sh): build the two HIP modules once per image,
# apply and verify the overlay, then exec `vllm serve`. Inputs are environment variables set by
# serve.sh; mounts: /overlay (this directory), /patches (repo root), /r4d-src (libr4d), /build,
# /cache, /models.
set -euo pipefail
SP=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
echo "[vnext] site-packages: $SP"
# The image's pip ROCm SDK: hipcc finds the HIP runtime through ROCM_PATH.
export ROCM_PATH="${ROCM_PATH:-${SDK_DEV:-/opt/rocm}}"
if [ ! -f /build/radiance_mxfp4_fp8.so ]; then
  echo "[vnext] building radiance_mxfp4_fp8.so"
  ( cd /build && hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      /patches/radiance_mxfp4_fp8.hip -o /build/radiance_mxfp4_fp8.so.tmp && mv radiance_mxfp4_fp8.so.tmp radiance_mxfp4_fp8.so )
fi
if [ ! -f /build/r4d.so ]; then
  echo "[vnext] building r4d.so from /r4d-src"
  rm -rf /build/r4d-src && cp -r /r4d-src /build/r4d-src
  ( cd /build/r4d-src && rm -f ./*.o r4d.so && GFX_ARCH=gfx1201 OUT=/build/r4d.so.tmp ./build.sh && mv /build/r4d.so.tmp /build/r4d.so )
fi
cp /build/radiance_mxfp4_fp8.so /build/r4d.so "$SP"/
# aiter's fallback GEMM configs for gfx1201 (unused while every layer is on the radiance kernel)
CFG="$SP/aiter/ops/triton/configs/gfx1201/triton/gemm/gemm_afp4wfp4"
mkdir -p "$CFG"
cp /patches/mxfp4-configs/gfx1201-GEMM-AFP4WFP4.json "$CFG/DEFAULT.json"
for f in /patches/mxfp4-configs/gfx1201-GEMM-AFP4WFP4-N=*.json; do cp "$f" "$CFG/$(basename "$f" | sed 's/^gfx1201-//')"; done
cd /
python3 /overlay/apply_overlay.py | grep -v '^  \(BASE\|NEW\) ' || true
python3 /overlay/verify_overlay.py
if [ "${PREPARE_ONLY:-0}" = 1 ]; then echo "[vnext] prepared, no serve"; exit 0; fi

ARGS=()
[ -n "${SPEC:-}" ] && ARGS+=(--speculative-config "$SPEC")
[ "${TOOLS:-1}" = 1 ] && ARGS+=(--enable-auto-tool-choice --tool-call-parser qwen3_coder)
# shellcheck disable=SC2086  # SERVED_NAMES and EXTRA are word lists on purpose
exec vllm serve "/models/$MODEL" --served-model-name $SERVED_NAMES --host "$HOST" --port "$PORT" \
  --kv-cache-dtype fp8 --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --max-num-batched-tokens "$CHUNK" \
  --attention-backend CUSTOM "${ARGS[@]}" --no-async-scheduling \
  --compilation-config '{"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true}}' \
  --enable-prefix-caching --enable-prompt-tokens-details --reasoning-parser qwen3 \
  --override-generation-config "$GEN" --chat-template "/patches/$CHAT_TEMPLATE" ${EXTRA:-}
