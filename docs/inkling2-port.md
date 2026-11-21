# Inkling 2 Mini on R9700 (gfx1201) — port plan

Status: **started 2026-11-21** (integration branch built, nothing runs on gfx1201 yet).
Owner: this fork. Decision: we do **not** wait for upstream vLLM to merge the Inkling 2 Mini
and ROCm PRs — we carry them ourselves and port to RDNA4.

## Why

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines,
Apache-2.0, 2026-11-20): 26.4B total / 3.4B active looped MoE (5.4B effective active with three
passes), 1M context, text + image. Its model card puts it level with Qwen3.8-27B overall
(AA Index 20.1 both) and ahead on agentic search, long context and safety — at a fraction of the
per-token compute, so it should decode much faster than the dense 27B we serve today.

## Model facts that matter for the port

| | |
|---|---|
| Architecture | `Inkling2LoopForConditionalGeneration` (`model_type: inkling2_loop`) |
| Layers | 24 decoder layers; **layers 6–17 run 3 passes** with shared weights (48 layer applications per token); every pass has its own KV and conv-state caches; from pass 2 on, gated messages from the previous pass are merged into each core sublayer input (`gate_rank 64`) |
| Attention | hybrid: sliding window 1024 + global layers, relative-position term (`d_rel 16`, `rel_extent 1024`), qk L2 clipping, attention-residual blocks (`attn_res`, 6 layers per block), short conv (`sconv_kernel_size 4`) |
| MoE | 128 routed experts, top-8, no shared expert, `intermediate_size 1152`, gate bias, gate/up softcaps |
| Hidden | 2304, 24 heads (8 KV), head_dim 128, vocab 201024 |
| Checkpoint | ~20 GB, one safetensors: experts **NVFP4** (U8 packed + FP8-E4M3 per-16 scales), rest **MXFP8** + BF16 (`quant_method: modelopt`, `hf_quant_config.json`) |
| Extras | `mtp.safetensors` = DSpark speculator (3 layers, markov head); vision tower (23 layers) |

## What exists upstream (2026-11-21)

| PR | What | State |
|---|---|---|
| vllm#67325 | Inkling 2 Small (NVIDIA) | merged 2026-11-18 |
| vllm#67880 | Inkling 2 Hopper (SM90) — moved code to `vllm/models/inkling2/nvidia/` | merged 2026-11-19 |
| vllm#68761 | Hopper Humming expert GEMMs | merged 2026-11-19 |
| vllm#69028 | **Inkling 2 Mini** (`nvidia/loop.py`, CUDA `csrc/inkling2/conv_mix.cu`) | open |
| vllm#67685 | Inkling 2 **Small on ROCm**, `vllm/models/inkling2/amd/` (Triton, ~6.1k lines), **gfx950 / MI355X only**, all-MXFP8 checkpoint | open |
| llama.cpp #32187, #32695 | Inkling 2 / Mini in llama.cpp | open |

No ROCm implementation of the **looped** Mini exists anywhere, and nothing targets RDNA4.

## Our integration base

`inkling2/vllm-main-bf151b7eb-inkling2.diff` in this repo = vLLM `main@bf151b7eb` +
merge of `pull/69028` (`fc1443152`) + cherry-pick of the ROCm commit of `pull/67685`
(`a6aa3ad9e`). Built in a local vLLM clone (branch `inkling2-rdna4`).

Conflict resolution done while integrating:

- `vllm/models/inkling2/__init__.py`: #67685 was written against the flat layout of #67325's first
  version; main later moved NVIDIA code to `nvidia/`. New `__init__` resolves the three exports
  (`Inkling2ForConditionalGeneration`, `Inkling2LoopForConditionalGeneration`,
  `Inkling2DSparkModel`) lazily from `nvidia/` or `amd/` by platform.
- `tests/models/inkling2/*`: kept main's (NVIDIA) versions; RDNA4 tests are written fresh (M2).
- Static check: every relative import in `amd/` resolves and every name it takes from the shared
  modules (`configs`, `processing`, `speculator_config`) still exists on main.

## Gaps for gfx1201

1. **Hard gates.** `amd/model.py` refuses anything but gfx950 (`on_gfx950()`), and
   `amd/quant_config.py` asks for capability 95.
2. **MXFP8 GEMMs use `tl.dot_scaled`** (`amd/ops/triton_bmm.py`, gfx950 scaled MFMA). RDNA4 has
   FP8 (OCP e4m3) WMMA but no scaled MMA: split K into 32-wide blocks and apply the E8M0 scale
   products to the partial accumulators, or reuse radiance's fp8-WMMA kernels
   (`launch6_*` in `radiance_mxfp4.py` / `radiance_mxfp4_fp8.hip`). Tile/stage choices assume gfx950's 160 KB LDS
   (`amd/ops/linear.py`); gfx1201 allows 64 KB of LDS per workgroup and runs wave32 — every `num_warps`/`num_stages`
   needs retuning.
3. **NVFP4 experts.** The ROCm PR only runs the all-MXFP8 Small checkpoint. Mini ships NVFP4
   experts. Options, in order of preference:
   - a) **NVFP4 → MXFP4 at load** (radiance already has `radiance_nvfp4.py` /
     `patch_nvfp4_mxfp4.py`) and run the experts on radiance's MXFP4 MoE kernels — smallest
     (≈ 0.5 byte/param) and our fastest path; lossy (per-16 E4M3 scales → per-32 E8M0), so
     quality must be measured.
   - b) NVFP4 → FP8-E4M3 per-16-block dequant for the fp8-WMMA path (≈ 2× the memory of a).
   - c) a native NVFP4 Triton kernel (dequant in registers) — most work.
4. **The loop.** There is no `amd/loop.py`. Port `nvidia/loop.py` (403 lines) and
   `nvidia/ops/loop.py` (132) onto the `amd/` building blocks; the CUDA kernels
   `csrc/inkling2/conv_mix.cu` and `allreduce_conv_mix.cu` become Triton (or HIP) kernels.
   Per-pass KV/conv caches: 12 core layers × 3 passes — check the KV-cache spec/allocator
   accepts this on our side.
5. **vLLM base vs radiance.** Radiance's maintained overlay targets vLLM 0.32 (`v032/`); Inkling 2
   lives on main (0.35-dev). Two tracks:
   - **Track A (first):** a standalone gfx1201 ROCm image of the integration base, Inkling only,
     no radiance patches (TP=1 — the model fits one 32 GB card).
   - **Track B (later):** move the radiance overlay to the new base (all-reduce, GDN, drafthead…)
     so one image serves both Qwen3.8 and Inkling, then TP=2 if it pays off.
6. **Later:** DSpark speculator (`mtp.safetensors`), vision tower, tool/reasoning parsers are in
   the PR already (`--reasoning-parser inkling`).

## Milestones

| | Goal | Done when |
|---|---|---|
| M0 | Integration base + this plan | ✅ 2026-11-21 |
| M1 | Reference: HF transformers (version with `inkling2_loop`) on CPU, bf16, ~250 GB RAM; dump logits/hidden states for a fixed prompt set | reference files + script committed |
| M2 | gfx1201 build of the integration base (Track A image); gates lifted; unit tests for norm/sconv/attn_res/attention/moe ops at gfx1201 tolerances | `pytest tests/models/inkling2 -k rdna4` green |
| M3 | `amd/loop.py` + conv_mix kernels; text-only, eager, TP=1, experts via path 3b (simplest correct) | greedy output matches M1 (top-1 agreement ≥ 99 %, logit MSE logged) |
| M4 | Experts via NVFP4→MXFP4 (3a) on radiance kernels; CUDA/HIP graphs; tuned tiles | perplexity within agreed bound of M1; tok/s ≥ Qwen3.8 radiance |
| M5 | Serving: `serve` script, API gateway endpoint, DSpark, vision | in use behind an API gateway |

## Rules

- GPU work needs the radiance container stopped: only in windows the owner agrees to.
- Re-sync with upstream PRs as they change (they are open); keep our changes as separate commits
  on top so rebasing stays mechanical.
- Measure before claiming: every speed or quality number in this file comes with the command
  that produced it.
