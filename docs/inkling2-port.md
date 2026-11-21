# Inkling 2 Mini on R9700 (gfx1201) — port plan

Status: **M0 done 2026-11-21** (integration branch built; plan reviewed against the code the same day). Nothing runs on gfx1201 yet.
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
| M1 | **Reference.** transformers PR #51050 branch, CPU (~250 GB RAM, no GPU needed); fixed prompt set; dump logits + per-layer-visit hidden states; note whether activations are quantized | script + dumps committed (dumps outside git) |
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
