# vnext measurements (2026-11-21/22)

Two R9700, TP=2, DaisyRose-27B MXFP4 (W4A8), DFlash2-FP8 drafter with 7 draft tokens, fp8 KV,
262,144 context, 8 sequences, 8192-token chunks, GPU utilisation 0.90, embed-host on.
"production" = `serve-mxfp4.sh` on `stilldeadcode/vllm-radiance:0.9.3` (vLLM 0.27.1, libr4d rx6).
"vnext" = `vnext/serve.sh` on the image in `BASE` (libr4d rx9). One run each.

## Short prompts, one stream (`tests/specdec_compare.py`, 18 prompts x 512 tokens)

| | tok/s T=0 | tok/s T=1 | acceptance T=0 | acceptance T=1 |
|---|--:|--:|--:|--:|
| production | 156.0 | 137.4 | 4.30 | 3.89 |
| vnext | 158.8 | 143.6 | 4.28 | 4.01 |
| vnext, DSpine-b16 drafter (fp16 state cache) | 122.3 | 109.8 | 4.96 | 4.58 |

## Context length and parallel sessions (`bench_ctx.py`, 600 output tokens)

Prefill: all sessions start together, prompts never seen before. Decode: the same prompts again,
served from the prefix cache. Values are production / vnext.

| context | sessions | first token, cold (s) | prefill tok/s, all sessions | decode tok/s per session | decode tok/s total | first token, cached (s) |
|--:|--:|--:|--:|--:|--:|--:|
| 1,000 | 1 | 0.31 / 0.32 | 2962 / 2921 | 183.9 / 160.0 | 184 / 160 | 0.32 / 0.32 |
| 1,000 | 2 | 0.48 / 0.48 | 2876 / 2910 | 118.7 / 120.5 | 219 / 227 | 0.49 / 0.48 |
| 1,000 | 3 | 0.73 / 0.74 | 2983 / 2925 | 97.5 / 100.2 | 263 / 273 | 0.73 / 0.75 |
| 8,000 | 1 | 2.49 / 2.55 | 3188 / 3114 | 134.7 / 136.0 | 135 / 136 | 0.95 / 0.45 |
| 8,000 | 2 | 4.35 / 4.78 | 3285 / 3163 | 94.9 / 106.2 | 173 / 190 | 1.43 / 0.68 |
| 8,000 | 3 | 5.77 / 6.34 | 3310 / 3183 | 85.5 / 87.8 | 200 / 234 | 2.19 / 1.04 |
| 32,000 | 1 | 10.2 / 10.9 | 3122 / 2938 | 116.6 / 119.3 | 117 / 120 | 0.84 / 0.29 |
| 32,000 | 2 | 16.5 / 17.5 | 3139 / 2956 | 104.8 / 97.7 | 191 / 187 | 1.24 / 0.42 |
| 32,000 | 3 | 22.1 / 23.4 | 3148 / 2968 | 76.5 / 87.7 | 189 / 231 | 1.92 / 0.51 |
| 128,000 | 1 | 48.0 / 54.0 | 2665 / 2369 | 109.5 / 103.0 | 110 / 103 | 1.44 / 1.15 |
| 128,000 | 2 | 72.7 / 82.1 | 2668 / 2369 | 76.1 / 80.4 | 139 / 149 | 2.09 / 1.53 |
| 128,000 | 3 | 97.3 / 109.1 | 2670 / 2376 | 61.7 / 62.8 | 151 / 162 | 3.16 / 2.26 |
| 250,000 | 1 | 112.3 / 134.1 | 2226 / 1864 | 100.0 / 102.8 | 100 / 103 | 2.04 / 0.87 |
| 250,000 | 2 | 169.5 / 202.2 | 2230 / 1869 | 60.9 / 66.1 | 111 / 127 | 2.93 / 1.19 |
| 250,000 | 3 | 225.8 / 269.4 | 2234 / 1868 | (does not fit) | (does not fit) | 227 / 49 |

- Decode: equal or better at every context except the single 1,000-token stream (one run; the
  short-prompt table above shows no gap).
- First token from the prefix cache: about half of production's.
- Cold prefill is slower on long contexts: -2..-4% at 8K, -6% at 32K, -11% at 128K, -16% at 250K.
  Not the drafter (2417 tok/s at 128K without one) and not the libr4d revision (2380 with rx6).
  Cause not found yet.
- Three sessions of 250,000 tokens do not fit either pool; vnext kept two of the three in cache,
  production had to prefill again.

## KV pool at 262,144 context

| | tokens | concurrency |
|---|--:|--:|
| production | 839,958 | 3.20x |
| vnext | 707,649 | 2.70x |
| vnext without a drafter | 1,022,361 | 3.90x |

vnext has more KV memory than production (16.8 against 15.4 GiB per card) but a block costs about
30% more once the drafter's layers are in the pool. Not analysed further.

## Prefix cache per turn of a growing conversation (`tests/hitdelta.py`)

Tokens that were already processed in the previous turn and are prefilled again: production
1,800-3,200 per turn, vnext 50-1,500 (state cache fp32 in both). With
`--mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype float16` vnext re-prefills 60-870 and the pool
grows by about 9%; the effect of the 16-bit state on answers was not measured.

## Correctness checks that pass on vnext and on production

`tests/cachecorrect.py` (20,000 tokens), `tests/cachecorrect2.py` (6,000 and 40,000 tokens, five
variants in a row), `tests/functest.py` (tool call, tool result, image, JSON schema, streamed
reasoning, thinking off).
