# vnext: radiance on a current vLLM

The radiance kernels (MXFP4 W4A8 GEMM, R4D attention and gated-delta-net, fp8 stream, radiance
all-reduce, DFlash2 draft and verify heads, embed-host) as an overlay on a current vLLM build,
for two Radeon AI PRO R9700 (gfx1201) at TP=2.

- Base: see `BASE` (vLLM commit and the image it was tested on).
- Derived from `../v032/` (the same overlay on vLLM 0.32.0). `v032/` is unchanged.
- The production serve (`../serve-mxfp4.sh`, image `stilldeadcode/vllm-radiance:0.9.3`,
  vLLM 0.27.1) is not touched by anything here.

## Serve

```bash
./serve.sh                                   # DaisyRose-27B + DFlash2-FP8, port 8080
PORT=8090 NAME=radiance-vnext-test SERVED_NAMES=test ./serve.sh   # test instance
PREPARE_ONLY=1 DETACH=0 ./serve.sh           # build kernels, apply + verify overlay, no GPU
```

`serve.sh` lists its environment variables at the top. It starts the image with
`entrypoint.sh`, which builds `radiance_mxfp4_fp8.so` and `r4d.so` once per image (kept under
`~/.cache/radiance-vnext/<image key>/build`), applies the overlay with `apply_overlay.py`,
checks it with `verify_overlay.py` and runs `vllm serve`. Both GPUs must be free.

`radiance-tp2.env` holds the radiance switches. They are the values the production launcher
gives its container at TP=2, plus two lines that this vLLM/ROCm needs:

| Line | Why |
|---|---|
| `HSA_ENABLE_IPC_MODE_LEGACY=0` | The image sets 1. Legacy GPU memory sharing needs the out-of-tree amdgpu driver; on a mainline kernel `hipIpcGetMemHandle` fails with "invalid argument", which breaks RCCL's direct transfers and radiance's all-reduce. With 0 the dmabuf path is used and both work. |
| `VLLM_ROCM_USE_AITER_CUSTOM_AR=0` | vLLM now fuses the all-reduce into the following norm itself when aiter's custom all-reduce is on. Radiance's fp8 stream does the same job with its own kernels; the two cannot both own the partial sums. With 1 the stream declines (logged) and vLLM's scheme runs. |

One line is easy to lose because it is not a `RADIANCE_*` or `VLLM_*` variable: `R4D_ATTN_FP8=3`.
libr4d reads it itself; it selects the 8-bit QK and PV legs of the prefill attention. Without it the
16-bit legs run and cold prefill is 11% slower at 128K tokens and 16% slower at 250K.

### KV cache size

`serve.sh` passes `--kv-cache-memory` (`KV_MEM`, default 15.4 GiB per card: what the production
serve gets from `GPU_UTIL=0.90` on vLLM 0.27). Left to itself this vLLM sizes the cache from the
memory it measures after warm-up and, on ROCm, keeps 3 GiB plus the activation peak free
(`KV_CACHE_MARGIN_FLOOR_BYTES` in `vllm/v1/worker/extensible_kv_cache.py`): 13.2 GiB, 723K tokens
instead of 846K. With 15.4 GiB the fuller card peaked at 31.4 of 32.6 GiB under eight 32K-token
prefills and three 250K-token sessions. If a card also drives a display or something
else takes VRAM there, lower `KV_MEM` or set `KV_MEM=` to get vLLM's own sizing back.

### Long prompts next to decoding sessions

Stock scheduling gives a decoding session one step per prefill step, so it nearly stops while
another session prefills a long prompt (0.6 instead of 31 stream chunks per second for two
minutes at 250K). `serve.sh` therefore passes `--long-prefill-token-threshold` (`BUSY_CHUNK`, 1664)
and `--prefill-schedule-interval` (`PREFILL_EVERY`, 8), and `radiance-tp2.env` sets
`RADIANCE_BUSY_PREFILL_MIN_TOKENS=3328`, a radiance rule in `scheduler.py`: prefills with at most
that many tokens left are exempt, and the cap applies only while something is decoding. Numbers
and the trade-off are in `RESULTS.md`.

## What changed against v032

Overlay files that needed more than a three-way merge:

| File | Change |
|---|---|
| `vllm/v1/core/kv_cache_planning.py` | The KV group-size search moved here from `kv_cache_utils.py` (vLLM moved the policy). |
| `vllm/_aiter_ops.py`, `kernels/linear/mxfp4/aiter.py` | The aiter fallback GEMM moved; the gfx12x enablement follows it. |
| `vllm/model_executor/layers/linear.py` | `apply_flattened` passes radiance's `(q, scale)` tuples through. |
| `vllm/v1/core/sched/scheduler.py` | Busy-prefill rule (`RADIANCE_BUSY_PREFILL_MIN_TOKENS`), see above. |
| `vllm/v1/worker/gpu/model_runner.py`, `radiance_verifyhead.py` | The fast verify head is disarmed right after the sampling step and checks the row count, so prompt-logprob and warm-up calls stay on the exact head. |
| `radiance_gdnmerge.py` | On this vLLM the merged projection feeds the standard core op, like `forward_cuda`. |
| `radiance_arnq.py` | Declines when vLLM's fused all-reduce+norm is active. |
| `radiance_v032_plugin/__init__.py` | The DFlash FP8 exclusion hook follows `map_draft_quant_config`. The R4D GDN layer declares that it cannot export prefill checkpoints (`_checkpoint_alignment = None`): vLLM's own prefill kernels now write the prefix-cache state checkpoints mid-prefill, the R4D prefill does not, and without this every prefix-cache hit restored stale state and produced wrong tokens. With it the scheduler splits the prefill at block boundaries, as before. `RADIANCE_GDN_DECLINE=prefill\|decode\|all` hands those steps to vLLM's core for diagnosis. |
| `model_loader/base_loader.py`, `radiance_embed_host.py` | embed-host from the production fork. |

Not carried over:

- The top-k "composite" sampler change (3 files): vLLM rewrote that code. Sampling runs stock.
- `radiance_draft.py` (dynamic draft length) hooks the removed model runner v1; it installs
  nothing here, as on 0.32.
- Removed vLLM options: `--mamba-cache-mode` (prefix caching implies it),
  `disable_padded_drafter_batch`.

## Regenerating the overlay

The overlay is generated from a vLLM worktree that carries the radiance changes as one or more
commits on top of the base commit:

```bash
python3 ../v032/step1/build_overlay.py <vllm worktree> <base commit> .
```

It rewrites `files/`, `base/`, `patched/`, `diffs/` and `manifest.tsv` (rename the two files in
`diffs/` to the base commit). To move to a newer vLLM: rebase or three-way merge the worktree
onto the new commit, fix what breaks, regenerate, update `BASE`, run the tests below.

## Tests

Against a running server (`tests/`, plain Python, no dependencies):

| Script | Checks |
|---|---|
| `cachecorrect.py <url> <model> [tokens]` | The same long prompt cold, from the prefix cache, and with a new question after the cached prefix must all answer correctly. |
| `cachecorrect2.py <url> <model> <tokens> <first_max> <thinking> <temp>` | As above, but the first request stops after `first_max` tokens. Run both several times in a row on one server after any change to the GDN path, KV grouping, drafter or cache settings: a wrong cache shows only from the second request on, and a server that has just started can pass by chance. |
| `functest.py <url> <model>` | Tool call and tool result, image input, JSON schema output, streamed reasoning, thinking off. |
| `specdec_compare.py <label> <model> <out.json> [url]` | Speed and acceptance length on 18 short prompts, greedy and sampled. |
| `hitdelta.py <url> <model>` | Tokens prefilled again per turn of a growing conversation. |
| `soak.py <url> <model> <minutes> [workers]` | Several clients send a random mix (record lookups over shared documents, growing conversations, tool calls, JSON schema, long streams with one in three aborted mid-stream) and every checkable reply is checked. |

`tests/check_symbols.py` and `tests/selftest_overlay.py` are the static checks from v032.

## Measurements

See `RESULTS.md`.

## Drafters other than DFlash2

Pass a full `SPEC`. The drafter must name its own attention backend
(`"attention_backend":"TRITON_ATTN"`): the R4D backend is built for the target's head shape and
aborts the GPU queue on another one.

- DSpine (`tencent/Qwen3.8-27B-DSpine-b16`): `"method":"dflash"`, exactly 15 draft tokens,
  `"draft_tensor_parallel_size":2`. Works. vLLM's adaptive verification does not: it needs an
  attention backend that accepts varying query lengths, which the R4D backend does not declare.
