# vnext measurements

## 2026-11-23: base moved to vLLM v0.34.0, and a slow PCIe link found

**Everything from 2026-11-21 to 2026-11-23 morning below this section was measured with the
second card's link degraded.** Its root port had trained at PCIe 2.0 x8 (5 GT/s) instead of 5.0
x8 (32 GT/s) at a boot on 2026-11-19 (the first card was fine; both read back 14-21 GB/s on
2026-10-06). A reboot retrained it. At TP=2 the cards exchange activations every layer, so the
slow link capped prefill: the absolute prefill figures further down are low, while each
comparison in them is fair (both sides had the same link). Check a box with

    for p in /sys/bus/pci/devices/*/; do [ -f $p/current_link_speed ] && echo "$p $(cat $p/current_link_speed) x$(cat $p/current_link_width)"; done

under load (an idle card drops its link to save power) and look for a GPU's upstream port well
below its maximum.

The overlay now targets the v0.34.0 release (`BASE`; README "Port to v0.34.0"): the nightly tag it
was built on would be deleted after ~5 weeks. Same gate on both, both on the fixed links,
`tests/onecard-gate.sh` with `CARD=0,1`, `SIZES=1000,8000,32000,128000,250000`, `SESSIONS=1,2,3`
and the vLLM source tree as corpus; values are nightly / v0.34.0, one run each.

| | nightly | v0.34.0 |
|---|---|---|
| functional checks, answers after a cache hit, 20-minute soak | pass | pass |
| KV pool | 845,693 tokens | 841,218 tokens |

| context | sessions | prefill tok/s, all sessions | decode tok/s, total |
|--:|--:|--:|--:|
| 1,000 | 1 | 3830 / 3782 | 230 / 167 |
| 1,000 | 2 | 5119 / 5113 | 255 / 267 |
| 1,000 | 3 | 5186 / 5167 | 346 / 364 |
| 8,000 | 1 | 5592 / 5609 | 148 / 143 |
| 8,000 | 2 | 5621 / 5650 | 282 / 228 |
| 8,000 | 3 | 5630 / 5660 | 292 / 291 |
| 32,000 | 1 | 5160 / 5182 | 122 / 126 |
| 32,000 | 2 | 5179 / 5205 | 251 / 227 |
| 32,000 | 3 | 5190 / 5214 | 297 / 294 |
| 128,000 | 1 | 3955 / 3971 | 114 / 125 |
| 128,000 | 2 | 3970 / 3988 | 185 / 180 |
| 128,000 | 3 | 3975 / 3995 | 197 / 204 |
| 250,000 | 1 | 3023 / 3038 | 101 / 106 |
| 250,000 | 2 | 3034 / 3050 | 141 / 138 |
| 250,000 | 3 | 3035 / 3052 | 161 / 160 |

Single stream (`specdec_compare.py`): T=0 171.1 / 171.3 tok/s, T=1 155.5 / 157.6; acceptance T=0 4.16 / 4.17.

Prefill is the same within 0.5%. Decode differs by more than the run-to-run noise in two cells (1K x 1
and 8K x 2), in opposite directions in the others; the 1K decode phase is short. Read: the same
engine speed, with a release image.

### One card on v0.34.0

Same gate, card 1, values are 0.9.3 (`serve-tp1.sh`, measured that morning on the slow link) /
v0.34.0 (fast link). Cells whose sessions do not fit the cache together are left out (they prefill
again and their decode figure means nothing).

| | 0.9.3 | v0.34.0 |
|---|---|---|
| functional checks, answers after a cache hit | pass | pass |
| 20-minute soak | 1 wrong answer (60K-token lookup) | pass |
| KV pool (`KV_MEM` 8.5 GiB) | 248,235 tokens | 247,052 tokens |

| context | sessions | prefill tok/s | decode tok/s, total |
|--:|--:|--:|--:|
| 1,000 | 1 | 3299 / 2735 | 100 / 101 |
| 1,000 | 2 | 3221 / 2888 | 207 / 156 |
| 1,000 | 3 | 3293 / 2986 | 235 / 234 |
| 8,000 | 1 | 3337 / 3343 | 130 / 82 |
| 8,000 | 2 | 3340 / 3374 | 175 / 141 |
| 8,000 | 3 | 3377 / 3385 | 237 / 184 |
| 32,000 | 1 | 3028 / 2999 | 95 / 80 |
| 32,000 | 2 | 3048 / 3022 | 133 / 128 |
| 32,000 | 3 | 3053 / 3022 | 173 / 160 |
| 128,000 | 1 | 2199 / 2187 | 72 / 65 |
| 200,000 | 1 | 1820 / 1811 | 63 / 63 |

Single stream: T=0 106.0 / 93.5 tok/s, T=1 95.2 / 84.4.

Correct, the same cache and prefill from 8K up, but decode is 10-35% behind at one and two
sessions. One card therefore stays on 0.9.3 (`../up.sh` default). The int2 verify head did not arm
on one card at all before 2026-11-23 (arming ran out of memory and the error was swallowed);
fixing that took the nightly from 74.9 to 80.3 tok/s, the rest of the gap is not diagnosed.

## 2026-11-22 (nightly base, slow link on the second card)

Two R9700, TP=2, MXFP4 target (W4A8), DFlash2-FP8 drafter with 7 draft tokens, fp8 KV, 262,144
context, 8 sequences, 8192-token chunks, embed-host on, fp32 state cache.
"production" = `serve-mxfp4.sh` on `stilldeadcode/vllm-radiance:0.9.3` (vLLM 0.27.1, libr4d rx6,
`GPU_UTIL=0.90`). "vnext" = `vnext/serve.sh` on the image in `BASE` (libr4d rx9, `R4D_ATTN_FP8=3`,
`KV_MEM` 15.4 GiB). Values are production / vnext. One run each unless noted; two runs of the same
setup differed by up to 4% on the short-prompt test, so smaller differences mean nothing.

Targets: `abhinand/DaisyRose-27B-Quark-AWQ-MXFP4` and `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (MTP head
rewritten to FP8 with `fp8_mtp.py`).

## Short prompts, one stream (`tests/specdec_compare.py`, 18 prompts x 512 tokens)

| target | serve | tok/s T=0 | tok/s T=1 | acceptance T=0 | acceptance T=1 |
|---|---|--:|--:|--:|--:|
| DaisyRose | production (2 runs) | 156.0, 155.5 | 137.4, 139.0 | 4.30, 4.30 | 3.89, 3.88 |
| DaisyRose | vnext (3 runs) | 158.8, 151.9, 157.3 | 143.6, 137.5, 145.6 | 4.28, 4.18, 4.23 | 4.01, 3.85, 4.00 |
| Qwen3.8 | vnext | 157.0 | 143.0 | 4.25 | 3.95 |

## Context length and parallel sessions (`bench_ctx.py`, 600 output tokens)

Prefill: all sessions start together with prompts never seen before (the first vnext cell, 1,000 x 1,
is the first request after start-up and includes warm-up). Decode: the same prompts again,
served from the prefix cache.

### DaisyRose-27B

| context | sessions | first token, cold (s) | prefill tok/s, all sessions | decode tok/s per session | decode tok/s total | first token, cached (s) |
|--:|--:|--:|--:|--:|--:|--:|
| 1,000 | 1 | 0.31 / 0.38 | 2962 / 2409 | 183.9 / 177.3 | 184.2 / 177.6 | 0.32 / 0.33 |
| 1,000 | 2 | 0.48 / 0.48 | 2876 / 2916 | 118.7 / 117.2 | 218.5 / 208.5 | 0.49 / 0.48 |
| 1,000 | 3 | 0.73 / 0.74 | 2983 / 2933 | 97.5 / 96.8 | 262.9 / 270.6 | 0.73 / 0.75 |
| 8,000 | 1 | 2.49 / 2.52 | 3188 / 3150 | 134.7 / 130.7 | 134.9 / 130.9 | 0.95 / 0.46 |
| 8,000 | 2 | 4.35 / 4.73 | 3285 / 3200 | 94.9 / 107.8 | 173.1 / 203.5 | 1.43 / 0.68 |
| 8,000 | 3 | 5.77 / 6.26 | 3310 / 3226 | 85.5 / 83.4 | 200.1 / 205.3 | 2.19 / 1.05 |
| 32,000 | 1 | 10.23 / 10.57 | 3122 / 3020 | 116.6 / 110.5 | 116.8 / 110.7 | 0.84 / 0.27 |
| 32,000 | 2 | 16.5 / 16.91 | 3139 / 3057 | 104.8 / 97.9 | 190.8 / 185.7 | 1.24 / 0.42 |
| 32,000 | 3 | 22.12 / 22.8 | 3148 / 3051 | 76.5 / 85.9 | 188.7 / 240.4 | 1.92 / 0.48 |
| 128,000 | 1 | 48.0 / 49.19 | 2665 / 2601 | 109.5 / 105.1 | 109.7 / 105.3 | 1.44 / 0.98 |
| 128,000 | 2 | 72.69 / 74.8 | 2668 / 2602 | 76.1 / 83.0 | 138.9 / 151.4 | 2.09 / 1.46 |
| 128,000 | 3 | 97.29 / 99.55 | 2670 / 2604 | 61.7 / 61.9 | 151.4 / 163.0 | 3.16 / 2.0 |
| 250,000 | 1 | 112.29 / 115.58 | 2226 / 2162 | 100.0 / 99.7 | 100.2 / 99.9 | 2.04 / 0.92 |
| 250,000 | 2 | 169.46 / 173.98 | 2230 / 2176 | 60.9 / 62.6 | 111.1 / 121.4 | 2.93 / 1.18 |
| 250,000 | 3 | 225.75 / 231.57 | 2234 / 2175 | (does not fit) / 72.2 | (does not fit) / 108.6 | 227.23 / 4.34 |

### Qwen3.8-27B

| context | sessions | first token, cold (s) | prefill tok/s, all sessions | decode tok/s per session | decode tok/s total | first token, cached (s) |
|--:|--:|--:|--:|--:|--:|--:|
| 1,000 | 1 | 0.33 / 0.32 | 2807 / 2939 | 155.7 / 202.5 | 155.9 / 202.8 | 0.31 / 0.32 |
| 1,000 | 2 | 0.48 / 0.48 | 2869 / 2915 | 118.1 / 123.2 | 208.6 / 223.4 | 0.49 / 0.48 |
| 1,000 | 3 | 0.74 / 0.74 | 2938 / 2931 | 96.9 / 100.8 | 256.8 / 268.0 | 0.74 / 0.75 |
| 8,000 | 1 | 2.47 / 2.53 | 3206 / 3128 | 133.9 / 121.7 | 134.1 / 121.9 | 0.95 / 0.45 |
| 8,000 | 2 | 4.35 / 4.74 | 3291 / 3192 | 97.8 / 101.1 | 174.9 / 186.5 | 1.42 / 0.68 |
| 8,000 | 3 | 5.74 / 6.31 | 3323 / 3202 | 78.8 / 86.7 | 196.0 / 229.4 | 2.24 / 1.02 |
| 32,000 | 1 | 10.14 / 10.6 | 3150 / 3013 | 125.6 / 106.9 | 125.8 / 107.0 | 0.84 / 0.27 |
| 32,000 | 2 | 16.39 / 17.0 | 3159 / 3041 | 100.5 / 90.2 | 171.7 / 176.6 | 1.25 / 0.4 |
| 32,000 | 3 | 21.97 / 22.83 | 3171 / 3045 | 76.6 / 83.2 | 190.8 / 236.0 | 1.9 / 0.48 |
| 128,000 | 1 | 47.76 / 49.35 | 2679 / 2592 | 109.7 / 113.8 | 109.9 / 114.0 | 1.45 / 0.9 |
| 128,000 | 2 | 72.34 / 74.92 | 2680 / 2598 | 84.6 / 78.4 | 148.0 / 140.8 | 2.09 / 1.32 |
| 128,000 | 3 | 96.92 / 99.85 | 2681 / 2599 | 59.2 / 65.6 | 140.7 / 172.7 | 3.15 / 1.87 |
| 250,000 | 1 | 111.9 / 115.38 | 2234 / 2166 | 99.6 / 96.0 | 99.7 / 96.2 | 2.06 / 0.75 |
| 250,000 | 2 | 168.75 / 174.1 | 2239 / 2174 | 64.6 / 66.2 | 116.3 / 122.6 | 2.94 / 1.01 |
| 250,000 | 3 | 225.19 / 232.48 | 2239 / 2167 | (does not fit) / 51.1 | (does not fit) / 141.8 | 151.57 / 1.46 |

- Cold prefill: vnext is 2-4% slower at every size. What it is not (measured 2026-11-22):
  - not the chunk size: 8320 tokens (five whole cache blocks per step instead of four) changed nothing;
  - not the compiler: `radiance_mxfp4_fp8.so` and `r4d.so` built with production's ROCm 7.14 hipcc and
    loaded in this image gave the same prefill (3018-3035 tok/s at 32K, 2604-2610 at 128K);
  - not the CPU side or the number of steps: with `RADIANCE_STEP_TRACE=1` on both serves the
    difference is in GPU time per token and is nearly constant over the context (64K prompt,
    full steps: production 0.304 -> 0.363 ms/token, vnext 0.314 -> 0.375).
  Without a drafter vnext's first full step costs 0.303 ms/token, so the drafter's context step
  (concatenate five hidden states, `fc`, K/V of five layers, for every prefill token) is about
  3.6% of prefill here. Whether production pays less for it could not be measured: production does
  not start without the DFlash drafter on this checkpoint (`SPEC_METHOD=mtp` fails in compile).
- Decode: equal for one stream, equal or faster with several.
- First token from the prefix cache: half of production's or less.
- Three sessions of 250,000 tokens fit vnext's pool and stay cached; production has to prefill one
  of them again.

## Four and eight short sessions (DaisyRose, two runs each)

| context | sessions | prefill tok/s, all sessions | decode tok/s total |
|--:|--:|--:|--:|
| 1,000 | 4 | 3139, 3151 / 3046, 3023 | 272.6, 286.9 / 290.1, 324.0 |
| 1,000 | 8 | 3292, 3290 / 3153, 3165 | 402.9, 360.6 / 380.3, 390.5 |
| 8,000 | 4 | 3332, 3331 / 3218, 3219 | 218.8, 220.3 / 264.7, 260.0 |
| 8,000 | 8 | 3361, 3356 / 3233, 3233 | 256.7, 258.5 / 331.8, 334.7 |

The production serve shortens the draft as the batch grows (`RADIANCE_DYNAMIC_DRAFT`); vnext does
not have that hook and is not slower without it. vLLM's own schedule
(`"num_speculative_tokens_per_batch_size":[[1,2,7],[3,4,6],[5,8,5]]` in `SPEC`) works with the R4D
backend and gave the same totals within run-to-run noise (8 x 1,000: 383, 393; 8 x 8,000: 352, 334),
so it is not set by default.

## A long prefill next to decoding sessions (`tests/measure/m.py stall`, 2026-11-22)

Two sessions decode from an 8K context; a third sends a cold prompt. "Decode" is stream chunks
per second of each decoding session (31 when nothing else runs); "longest gap" is the p99 / max
time between two chunks during the prefill.

| setting | prompt | prefill tok/s | decode chunks/s | longest gap (p99 / max) |
|---|--:|--:|--:|--:|
| no throttling (`BUSY_CHUNK=0 PREFILL_EVERY=1`), as production behaves | 250K | 2121 | 0.6 | 4.1 s / 4.1 s |
| `--prefill-schedule-interval 8` | 100K | 2384 | 3.5 | 2.8 s / 3.0 s |
| chunk 1664, every 4th step | 100K | 2106 | 5.5 | 0.8 s / 1.9 s |
| chunk 3328, every 8th step | 100K | 2093 | 5.3 | 1.5 s / 2.4 s |
| **chunk 1664, every 8th step (default)** | 100K | 1823 | 8.9 | 0.8 s / 0.8 s |

The default adds a radiance rule on top of vLLM's two options
(`RADIANCE_BUSY_PREFILL_MIN_TOKENS=3328` in `radiance.env`): a prefill with at most that many
tokens left is neither capped nor deferred, and the cap applies only while something is
decoding. Without the rule, three cold 32K prompts advanced in lockstep (mean first token 30.5 s
instead of 22.7 s) and every short turn waited for its cadence slot. With it: three cold 32K
prompts 22.7 s / 3063 tok/s (as without throttling), a lone request unchanged.
Cost: under a constantly decoding load long prompts take longer (8-minute soak with six clients:
record lookups over 2K-60K-token documents averaged 38 s instead of 8 s).

## Other measurements of 2026-11-22 (`tests/measure/m.py`)

- Draft acceptance after a prefix-cache hit (`hit`): unchanged (acceptance length cold 3.45, same
  prompt from the cache 3.45; other question on a cached prefix 3.68, cold 3.69).
- Step time by context (`steps`, 7 draft tokens): 24.7 ms at 1K, 25.7 at 32K, 28.7 at 128K, 32.6 at
  250K. The growth is the same with 3 draft tokens, so it does not scale with the verify rows.
- Production with `KVCACHE=ram KVCACHE_RAM_GIB=64` (`evict`, `evictcheck`): an evicted 250K prompt
  is back in 13-23 s instead of 115 s. 5 of 6 answers correct after a restore (4 of 4 cold): not
  enough to clear it. A stopped server leaves `/dev/shm/vllm_offload_*.mmap` (64 GiB) behind.

## KV pool at 262,144 context

| | KV memory per card | tokens | concurrency |
|---|--:|--:|--:|
| production (`GPU_UTIL=0.90`) | 15.4 GiB | 839,958 | 3.20x |
| vnext, `KV_MEM` default | 15.4 GiB | 845,693 | 3.23x |
| vnext, `KV_MEM=` (vLLM's measured sizing) | 13.2 GiB | 722,987 | 2.76x |

Peak VRAM with the default, under eight 32K-token prefills followed by three 250K-token sessions:
29.6 and 31.4 of 32.6 GiB. No allocation failure.

## Prefix cache per turn of a growing conversation (`tests/hitdelta.py`, 2026-11-21)

Tokens that were already processed in the previous turn and are prefilled again: production
1,800-3,200 per turn, vnext 50-1,500.

## Correctness

Pass on vnext with both targets, and on production: `tests/cachecorrect.py` (20,000 tokens),
`tests/cachecorrect2.py` (6,000 and 40,000 tokens, several in a row), `tests/functest.py`.

Soak (`tests/soak.py`, vnext, DaisyRose, 30 minutes, 6 clients): 556 requests (204 record lookups
over shared documents, 71 growing conversations, 72 tool calls, 44 JSON-schema outputs, 165 streams
of which about a third were aborted mid-stream), 0 wrong answers, 0 errors, server answering
afterwards, VRAM flat.

## Earlier measurements kept for reference

- Without `R4D_ATTN_FP8=3` (first vnext runs, 2026-11-21): cold prefill 2369 tok/s at 128K and 1864
  at 250K (production 2665 / 2226).
- DSpine-b16 drafter on vnext (fp16 state cache): 122.3 / 109.8 tok/s at T=0 / T=1 with acceptance
  4.96 / 4.58: accepts more per step, slower overall than DFlash2.
