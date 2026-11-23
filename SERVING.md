# Serving on one or two R9700s

The short path for a desktop or a small server with one or two AMD Radeon AI PRO R9700 cards: set it
up once, then start and stop the server with two commands. `README.md` has the full story and every
setting; this page is the day-to-day part.

| | One R9700 | Two R9700 |
|---|---|---|
| Engine | current vLLM with this repo's overlay (`vnext/serve.sh`, image in `vnext/BASE`), **not yet tested on one card**; `ONE_CARD=0.9.3` serves the measured 0.9.3 single-GPU profile (`serve-tp1.sh`) instead | current vLLM with this repo's overlay (`vnext/serve.sh`, image in `vnext/BASE`) |
| Context | 220,000 tokens | 262,144 tokens |
| Requests at once | 3 | 4 (`MAXSEQS`) |
| Speed | see [One card or two](#one-card-or-two) | about 1.6x one card when generating, 1.75x when reading a prompt |

Both serve the same checkpoint with the same DFlash2 drafter, on the same port, under the same model
name, so a client does not notice which one is running.

## One card or two

The same benchmark on both: BetterBench, Qwen3.8-27B native MXFP4, DFlash2 drafter with 7 draft
tokens, the 0.9.3 engine on both (README.md, "Performance": two cards measured 2026-08-30, one card
2026-09-16 at a 210 W power cap with 65,536 context). The current-vLLM two-card serve that `up.sh`
starts generates as fast as 0.9.3 and reads prompts 2-4% slower (vnext/RESULTS.md), so the two-card
column holds for it within that margin.

| | One R9700 | Two R9700 | Two / one |
|---|--:|--:|--:|
| Generation, one stream (tok/s, mixed tasks) | 137.7 | 224.3 | 1.63x |
| Time per stream update (ms) | 35.2 | 22.9 | 0.65x |
| First token, short prompt (ms) | 95 | 53 | 0.56x |
| Total generation, 2 requests at once (tok/s) | 207.6 | 302.6 | 1.46x |
| Total generation, 4 requests at once (tok/s) | 319.0 | 442.7 | 1.39x |
| Total generation, 8 requests at once (tok/s) | 416.3 | 572.8 | 1.38x |
| Reading a prompt of ~6K tokens (tok/s) | 2,720 | 4,773 | 1.75x |
| Reading a prompt of ~47K tokens (tok/s) | 2,480 | 4,330 | 1.75x |
| KV cache (tokens, all requests together) | ~200,000 | ~940,000 | 4.7x |

What that means in use:

- **One person, one conversation:** one card is comfortable. 138 tok/s is faster than anyone reads;
  a 47K-token prompt takes ~19 s on one card against ~11 s on two.
- **Agents and long contexts:** two cards. An agent sends long prompts again and again; what is
  cached is not read again, and the KV cache decides how much stays cached: one card holds about one 200K-token
  session, two cards three of 250K.
- **Several users:** the gap narrows to ~1.4x in total throughput, but each request on one card is
  slower (79 against 104 tok/s per request at 8 at once).

The numbers are not from one sitting, and prefill on one card follows its power limit (3,552 tok/s
at 2K tokens with 300 W instead of 210 W, README.md), so take the ratios as about right, not exact.

## 1. Once: set up

You need Linux with the `amdgpu` driver, Docker able to reach `/dev/kfd` and `/dev/dri` (`up.sh` and the two-card serve use Docker; Podman works only for `setup-mxfp4.sh` and the `serve-*.sh` launchers),
git, about 60 GiB of disk and a Hugging Face login only if a download asks for one.

In a clone of this repository:

```bash
./setup-mxfp4.sh                      # Qwen3.8-27B (AMD's MXFP4 release) + the DFlash2 drafter
MODEL=daisyrose ./setup-mxfp4.sh      # DaisyRose-27B instead (abhinand's MXFP4 release)
```

`setup-mxfp4.sh` checks the host, pulls the 0.9.3 image, downloads the checkpoint, makes it loadable
(AMD's release gets its MTP head rewritten to FP8, ~15 minutes; DaisyRose already ships it in FP8
and is only taken over) and downloads the drafter. Run it once per model. It puts everything in
`~/models` (`MODELS=/elsewhere ./setup-mxfp4.sh` to change that).

Then the settings of this host:

```bash
cp local.env.example local.env        # and edit: MODELS, MODEL, PORT, ...
```

`local.env` is not tracked by git; with no `local.env` the defaults are `~/models`, Qwen3.8, port 8080.

## 2. Every day: start and stop

```bash
./up.sh          # start, and wait until it answers
./down.sh        # stop, and show that the cards' memory is free again
```

`up.sh` counts the usable cards and starts the current-vLLM serve on two cards (or the first two of
more) or on one. On one card that serve is recommended but not tested yet (next section); if it does
not start or misbehaves, `ONE_CARD=0.9.3 ./up.sh` (or `ONE_CARD=0.9.3` in `local.env`) serves the
measured 0.9.3 single-GPU profile instead.
`CARDS=1 ./up.sh` runs on one card on a two-card host, for example to keep the second card free.
`./up.sh --print` shows the container command without starting anything.

The container is called `radiance`. While it runs:

```bash
docker logs -f radiance                               # the server's log
curl http://localhost:8080/v1/models                  # what it serves
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Hello!"}]}'
```

After `./down.sh` the log stays readable with `docker logs radiance` until the next start.

**The first start is slow.** Both engines compile kernels the first time (and after an update): the
two-card serve builds two kernel libraries and fetches libr4d into `~/.cache`, the one-card serve
builds libr4d. Allow 5-15 minutes; later starts take about two. `up.sh` waits up to 15 minutes and
prints the log if the server stops on the way.

## One card on the current vLLM: still to test

One card runs the 0.9.3 single-GPU profile carried over to the current vLLM: same settings, same
kernels, and it builds and passes the overlay checks. It has not served on one card yet, and the
speed figures for one card on this page are 0.9.3's. Before it counts as tested:

1. **Starts and answers** on one card: `vnext/tests/onecard-gate.sh` (needs one free card, ~3 hours)
   runs everything below on 0.9.3 and on vnext, one after the other on the same card.
2. **Correct:** the functional checks (tool call, tool result, image, streamed reasoning, JSON
   schema), the right answer after a prefix-cache hit with thinking on and off, and a 20-minute soak
   without a wrong answer or an error.
3. **KV cache:** the pool vLLM sizes at `GPU_UTIL` 0.95 is at least 0.9.3's (~200,000 tokens at
   220,000 context); then pin it as `KV_MEM` for one card, as two cards have.
4. **Speed:** decode per session within ~3% of 0.9.3 and prefill within ~5%, the bar the two-card
   port was held to (vnext/RESULTS.md).
5. **Long context:** a 200,000-token prompt prefills and is served from the cache on the next turn.

Until then 0.9.3 is the safe choice on one card (`ONE_CARD=0.9.3`).

## Switching models

Set `MODEL=daisyrose` (or `qwen3.8`) in `local.env`, then `./down.sh && ./up.sh`. The model name
clients ask for changes with it (`DaisyRose-27B` or `Qwen3.8`); `SERVED_NAMES` in `local.env`
keeps a fixed name instead. Sampling for requests that do not set their own: temperature 1.0 for
DaisyRose (RadixArk's recommendation with thinking on), 0.7 for Qwen3.8; `GEN` in `local.env`
changes it.

## Updating

```bash
./down.sh && git pull && ./up.sh
```

A pull that changes the overlay or the kernels makes the next start compile again (the slow first
start above). The engines keep their caches per image, so going back to an older commit reuses
what was built for it.

## When something is wrong

| What you see | What to do |
|---|---|
| `no checkpoint in …` | run `MODEL=<model> ./setup-mxfp4.sh` for that model, or fix `MODELS` |
| `it is already running` | `./down.sh` first |
| `port 8080 is already in use` | another server holds the port: stop it, or set `PORT` |
| `the server stopped while starting` | the log is printed above the message; out-of-memory at start usually means something else holds VRAM (`./down.sh` lists what each card uses) |
| a card is missing | `./gpu-detect.sh` shows which cards it sees and why one is skipped |
