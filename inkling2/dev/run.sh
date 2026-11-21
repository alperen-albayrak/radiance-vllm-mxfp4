#!/usr/bin/env bash
# Run a command in the upstream vLLM ROCm nightly with the files our integration tree changes
# bind-mounted over the installed package (python-only changes: nothing is rebuilt).
#   TREE=~/src/vllm GPUS=1 inkling2/dev/run.sh python3 -m pytest /tree/tests/models/inkling2/test_rdna4_mxfp4.py
set -euo pipefail
TREE=${TREE:?set TREE to the vLLM clone with the inkling2 branch}
WORK=${WORK:-$PWD/inkling2-work}      # mounted at /work (reference dumps, outputs)
IMAGE=${IMAGE:-vllm/vllm-openai-rocm:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b}
BASE=${BASE:-nightly-base}            # git ref of the commit the image was built from
GPUS=${GPUS:-1}                       # HIP_VISIBLE_DEVICES
PKG=/usr/local/lib/python3.14/dist-packages
mounts=()
while IFS= read -r f; do
  case "$f" in
    vllm/models/inkling2/*) ;;        # the whole package is mounted below
    vllm/*) [ -e "$TREE/$f" ] && mounts+=(-v "$TREE/$f:$PKG/$f:ro") ;;
  esac
done < <(git -C "$TREE" diff --name-only "$BASE" -- vllm; git -C "$TREE" ls-files --others --exclude-standard -- vllm)
mounts+=(-v "$TREE/vllm/models/inkling2:$PKG/vllm/models/inkling2:ro")
exec docker run --rm ${TTY:+-it} --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
  --group-add video --ipc host --network host -e HIP_VISIBLE_DEVICES="$GPUS" -e HF_HUB_OFFLINE=1 \
  -e PYTHONDONTWRITEBYTECODE=1 ${EXTRA_ENV:-} \
  "${mounts[@]}" -v "$TREE:/tree:ro" -v "$WORK:/work" \
  -v "$(cd "$(dirname "$0")/.." && pwd):/i2:ro" \
  --entrypoint /bin/bash "$IMAGE" -lc "$*"
