# Inkling 2 Mini on R9700 (gfx1201) — port plan

Status: **runs on one R9700 with MXFP4 experts (2026-11-21)** — M0–M4 functional; speed tuning (M5) and serving integration (M6) open.
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
| Checkpoint | 19.5 GB, one safetensors (`hf_quant_config.json`, `quant_method: inkling2`). **Only the MoE experts are quantized** (`w13_weight`, `w2_weight`): layers **0–19 NVFP4** (e2m1 codes, e4m3 scale per 16, `amax_denominator` scale format), layers **20–23 MXFP8** (e4m3, e8m0 per 32×32). Everything else is BF16 (attention, sconv, embed, vision) or FP32 (AttnRes and loop queries) |
| Activations | the expert GEMMs take **quantized activations**: NVFP4 per token (`fouroversix_mae` scale rule, global scale amax/1536) in layers 0–19, MXFP8 (e8m0 per 32) in 20–23. The reference computes with these, so "bf16 activations" is a deviation that must be measured, not assumed harmless |
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

## Result (2026-11-21)

Inkling 2 Mini serves from vLLM on a single R9700 (gfx1201), experts in **MXFP4**, the loop
ported to the ROCm (`amd/`) model.

**How it is built.** Upstream ROCm nightly `vllm/vllm-openai-rocm:nightly-f8b29fab…` (vLLM main
@ `f8b29fab4`, torch 2.13.0+rocm10.0.0, triton 3.8.0 — it runs on gfx1201 as shipped) with our
tree's Python files bind-mounted over the installed package (`inkling2/dev/run.sh`,
`inkling2/dev/serve.sh`); nothing is compiled. The tree is vLLM `f8b29fab4` + merge of #69028 +
the ROCm commit of #67685 + our two commits (`inkling2/patches/`; whole diff:
`inkling2/vllm-f8b29fab-inkling2-rdna4.diff`).

**What we changed (all under `vllm/models/inkling2/amd/` unless noted).**

| | |
|---|---|
| `ops/mxfp4.py` (new) | NVFP4 → fp32 → MXFP4 (e2m1 two per byte, ue8m0 per 32; per block the better of the no-clip exponent and the one below, by squared error); reference dequantizers |
| `ops/triton_bmm.py` | the expert GEMM takes MXFP4 weights too (`tl.dot_scaled(..., "e2m1")`), MXFP8 activations as before (W4A8); on RDNA4 128-wide K tiles, one stage (64 KB LDS) |
| `moe.py` | per-layer expert format; NVFP4 tensors (codes, e4m3 scales, per-expert amax, denominator) loaded as stored, requantized in `process_weights_after_loading`, relative error logged |
| `attention.py`, `sconv.py` | per-pass KV caches (`Inkling2AttentionPassCache`) and conv states; local layers accept separate q/k/v inputs; KV cache API of current main (`get_kv_cache_config`) |
| `ops/attn_res.py` | optional output norm (raw fp32 mixes for the loop's blends) |
| `ops/loop.py` (new), `loop.py` (new) | message controllers (Triton, from the NVIDIA ops) and `Inkling2LoopModel`: prelude → 3 core passes → coda, unfused |
| `model.py` | passes per decoder layer, expert format from the quant config, gate widened to RDNA4, loop model selection |
| `ops/moe.py`, `ops/activation.py` | column blocks no longer assume Small's widths (hidden 2304, intermediate 1152) |
| `quant_config.py` | removed: the shared `inkling2/quant_config.py` is used |
| `inkling2/__init__.py` | lazy exports resolve to `nvidia/` or `amd/` by platform |

**Checks.**

- `tests/models/inkling2/test_rdna4_mxfp4.py`: 5 passed on gfx1201 (e2m1 round trip, exact
  requantization of representable weights, error bound, MXFP4 GEMM vs fp32 reference for both
  expert shapes).
- NVFP4 dequantization equals the HF reference's bf16 experts (max rel. diff 2.3e-3 = bf16
  rounding; `w13` rows are interleaved gate/up as `ops/activation.py` expects).
- NVFP4 → MXFP4 relative Frobenius error on the real weights: **0.110 (w13), 0.112 (w2)** in
  every layer. This is the price of the decision for MXFP4; it is of the same size as a 4-bit
  quantization itself.
- Against the M1 reference (`inkling2/compare_ref.py`, greedy, token ids as prompt): the first
  token agrees on all 6 prompts, the greedy continuations agree for the first 24–64 tokens
  (one prompt all 64), 248/384 positions overall. Divergence starts late and stays on topic —
  what quantization noise does (MXFP4 weights + MXFP8 activations vs the reference's bf16); a
  wrong loop or cache would diverge at once. The 2227-token prompt (beyond the 1024 window,
  per-pass caches) behaves the same.
- Speed, single stream, FULL CUDA graphs, untuned: **29.9 tok/s** (600 tokens, twice). Model
  memory 18.1 GiB.

**Open.**

1. Quality gate: GSM8K (radiance's gate) and perplexity against the reference — the 11 %
   weight error must be shown harmless on a benchmark, not on six prompts.
2. Speed (M5): the GEMM goes through Triton's `dot_scaled` lowering with small tiles; the mixes
   and short convs are unfused; no speculative decoding. Candidates: radiance's W4A8 fp8-WMMA
   kernel for the expert GEMM, tile tuning, conv_mix fusion, DSpark.
3. TP=2, vision tower, tool/reasoning parsers under load: not tested yet.
4. Layers 20–23 stay MXFP8 (as shipped); requantizing them to MXFP4 would save ~2 GB.
5. Upstreaming: the branch is local (`rdna4`); patches here.

## Review of the first draft (2026-11-21, after reading the code)

What the first version of this plan got wrong or missed:

1. **The two CUDA kernels are not needed for correctness.** `csrc/inkling2/conv_mix.cu` and
   `allreduce_conv_mix.cu` fuse the short conv with the next AttnRes mix (and with the TP
   all-reduce). `nvidia/ops/conv_mix.py` is that fusion; unfused it is `short_conv` followed by
   `attn_res`, and both already exist in `amd/ops/`. They are a later speed item, not a blocker.
2. **`nvidia/loop.py` cannot be copied onto `amd/`.** #69028 also changed `nvidia/model.py`,
   `attention.py`, `moe.py` and `sconv.py` (+327 lines: `pass_idx`, per-pass `conv_states`,
   `_make_layer`, `conv_mix`). The `amd/` package was written against Inkling 2 Small and has none
   of that: one `conv_state` per layer, `Inkling2Model._mix`, no pass argument. The port is
   "add passes to the amd model", not "translate one file".
3. **`nvidia/ops/loop.py` is plain Triton** (`pair_rms_norm`, `blend_rms_norm`): portable as is.
4. **A reference is harder than "install transformers".** `inkling2_loop` is not in a transformers
   release (5.23.0); it is open PR huggingface/transformers#51050. M1 uses that branch, and must
   check whether it applies the activation quantization (if it does not, it is not the reference
   the checkpoint was trained/evaluated with).
5. **`amd/` refuses the Mini checkpoint twice**: gfx950 gate, and "MXFP8 experts only" per layer
   (`Inkling2DecoderLayer.__init__`).
6. vLLM main already knows the card (`on_gfx1201()`, `on_rdna4()` in `vllm/platforms/rocm.py`),
   so the gates can be widened cleanly instead of bypassed.
7. Memory: experts ≈ 10.2 GB (NVFP4) + 4.1 GB (MXFP8 + scales), BF16 ≈ 3.9 GB → the model fits one
   32 GB R9700 with room for the KV caches of 12 core layers × 3 passes. **TP=1 first.**

## llama.cpp trial (2026-11-21)

Before porting, the model was tried through llama.cpp (master + #32704, Vulkan/RADV on one R9700;
the HIP backend segfaults in `hipMemsetAsync` with the host's ROCm 10.1): Unsloth
`Mainline_Support` Q8_K_XL and Q4_K_XL. Small smoke tests passed (Turkish, reasoning, a compiling
Rust task, a tool call, a diagram read through mmproj). Speed: **24 tok/s (Q8), 26 tok/s (Q4)**
single stream — the reason for this port.

## Probe: upstream ROCm ops on gfx1201 (2026-11-21)

`inkling2/tests/probe_amd_ops.py`, run inside `stilldeadcode/vllm-radiance:0.9.3` (torch
2.11.0+rocm7.14, triton 3.6.0) on one R9700, ops loaded straight from the integration tree:

| op (unchanged gfx950 code) | result vs torch fp32 |
|---|---|
| `norm.rms_norm` | ok, max rel err 3.3e-03 (bf16 rounding) |
| `linear.linear_k_groups` | ok, 2.4e-03 |
| `attn_res.attn_res` | runs, finite |
| `mxfp8.mxfp8_quantize` | runs |
| `triton_bmm.bmm` (**`tl.dot_scaled`**, the MXFP8 expert GEMM) | **ok, 2.5e-03** |

So Triton lowers `tl.dot_scaled` on gfx1201 too (without scaled MFMA it decomposes it): the
MXFP8 expert path is **correct on RDNA4 as written**; only its speed is unknown. All eleven op
modules import except `sconv` (it imports `vllm.models.inkling2`, absent from the image's vLLM
0.27.1). Consequences:

- Gap 5 ("MXFP8 layers need a new kernel") is a speed item, not a correctness one.
- The simplest correct GPU path for the NVFP4 layers is **NVFP4 → MXFP8 at load** onto this same
  GEMM (near-lossless: 8-bit values, e8m0 per 32). Cost: experts grow to ~20 GB (1 byte/param),
  ~28 GB with the rest — too tight for one 32 GB card with KV caches, fine with the experts split
  over both cards (`ep_size` exists in `amd/moe.py`). That is M3; MXFP4 (decision below) is M4
  and brings it back to one card.

## Decisions (owner, 2026-11-21)

- Do not wait for upstream: we carry #69028 + #67685 ourselves (done, M0).
- **Experts: NVFP4 → MXFP4 at load, on radiance's MXFP4 kernels.** Lossy (e4m3 per 16 → e8m0 per
  32), so every step logs the per-layer relative error and is checked against the reference.
- **We port the loop to `amd/` ourselves.**

## Gaps for gfx1201

1. **Gates.** `amd/model.py` (`on_gfx950()`), `amd/quant_config.py` (capability 95) and the
   per-layer "MXFP8 only" check.
2. **Passes in the amd model.** Per-pass KV caches and conv states for the core layers
   (`amd/attention.py`, `amd/sconv.py`), `pass_idx` through attention/MoE, `_make_layer`.
3. **`amd/loop.py`.** `Inkling2LoopModel` on amd's `_mix`/`attn_res` API: prelude → 3 core passes
   (each restarts from the prelude state; the previous pass's blocks are the message bank) → coda;
   `Inkling2MessageController` + `blend` (4 streams: attn, k, v, mlp; rank 64); loop queries
   (`middle_cycle_cross_pass_queries`, `..._kv_queries`). Unfused: sconv, then mix.
4. **Experts in MXFP4.** Radiance's converter (`radiance_nvfp4.py`) handles *linear* layers in
   the compressed-tensors layout. Here the weights are **stacked expert tensors** in the
   `inkling2` (modelopt) layout and the GEMM is a **grouped** one (128 experts, top-8, rows sorted
   by expert — `amd/ops/moe.py`, `triton_bmm.py`). Needed: a) loader: dequant NVFP4 per expert →
   requant e2m1 + e8m0/32 (reuse the `mse` exponent rule); b) a grouped MXFP4 expert GEMM for
   gfx1201 — either radiance's W4A8 fp8-WMMA kernel driven per expert segment, or a Triton
   `bmm` that decodes e2m1 in registers; c) activation side: fp8 (W4A8) as radiance does, measured
   against the NVFP4-activation reference.
5. **MXFP8 layers 20–23** use `tl.dot_scaled` (gfx950 scaled MFMA). gfx1201 has fp8 WMMA but no
   scaled MMA: apply the e8m0 block scales to 32-wide partial accumulators, or requantize these
   four layers to MXFP4 too (radiance's default for FP8 layers) so one kernel serves all experts.
6. **Tuning.** Tile/warp/stage choices assume gfx950 (160 KB LDS, wave64); gfx1201 allows 64 KB of
   LDS per workgroup and runs wave32.
7. **Base.** Radiance's overlay targets vLLM 0.32 (`v032/`); Inkling 2 lives on main (0.35-dev).
   Track A: standalone gfx1201 image of the integration base, Inkling only, importing radiance's
   kernel modules as a plugin. Track B (later): move the whole overlay to the new base so one
   image serves Qwen3.8 and Inkling.
8. **Later:** conv_mix fusion, DSpark speculator (`mtp.safetensors`, BF16), vision tower, TP=2.

## Milestones

| | Goal | Done when |
|---|---|---|
| M0 | Integration base + plan | ✅ 2026-11-21 |
| M1 | **Reference.** ✅ 2026-11-21 — `inkling2/ref_dump.py` (transformers PR #51050 @ 6173a2f, torch 2.15 CPU, bf16) on 6 prompts (25–2227 tokens): outputs of all 48 layer visits (last 8 positions), final logits, 64 greedy tokens. Dumps (1.8 GB) kept outside the repository, summary `inkling2/ref-meta.json`. Two of the prompts (`turkish`, `long`) were replaced on 2026-11-23 and dropped from the summary; dump the reference again before comparing against them. **The HF reference dequantizes the experts to bf16 and does not quantize activations** — so it is the "ideal weights-as-stored" reference; the NVIDIA serving path (NVFP4/MXFP8 activations) is a second, slightly different target | done |
| M2 | **Loop on CPU in our tree.** `amd/loop.py` + passes, with torch fallbacks for the GEMMs (dequantized experts, no Triton): proves the control flow and weight loading | hidden states match M1 at every visit (bf16 tolerance) |
| M3 | **gfx1201 kernels.** Track A image; gates widened; amd Triton ops (norm, sconv, attn_res, attention) pass their unit tests on gfx1201; experts via the simplest correct GPU path | greedy tokens match M1 on the prompt set |
| M4 | **MXFP4 experts.** NVFP4→MXFP4 loader + grouped GEMM; layers 20–23 decided (scaled fp8 or MXFP4) | per-layer error logged; top-1 agreement and perplexity vs M1 reported; GSM8K gate like other radiance kernels |
| M5 | **Speed.** HIP/CUDA graphs, tile tuning, conv_mix fusion | tok/s reported next to Qwen3.8-radiance, same harness |
| M6 | **Serving.** serve script, gateway endpoint, DSpark, vision | in use behind an API gateway |

M2 is new: it separates "is the loop right" from "are the gfx1201 kernels right", and needs no
GPU time — the radiance container keeps serving while it is built.

## Rules

- GPU work needs the radiance container stopped: only in windows the owner agrees to.
- Re-sync with upstream PRs as they change (they are open); keep our changes as separate commits
  on top so rebasing stays mechanical.
- Measure before claiming: every speed or quality number in this file comes with the command
  that produced it.
