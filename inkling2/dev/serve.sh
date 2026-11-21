#!/usr/bin/env bash
# Serve Inkling 2 Mini from the integration tree (see run.sh). Logs to /work/serve.log.
#   inkling2/dev/serve.sh [extra vllm serve args]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec "$here/run.sh" "vllm serve /work/Inkling-2-Mini --served-model-name inkling-2-mini \
  --host 127.0.0.1 --port ${PORT:-8090} --tensor-parallel-size ${TP:-1} \
  --max-model-len ${MAXLEN:-32768} --max-num-seqs ${MAXSEQS:-4} \
  --reasoning-parser inkling $* 2>&1 | tee /work/serve.log"
