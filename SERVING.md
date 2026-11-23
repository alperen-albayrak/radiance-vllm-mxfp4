# Serving on one or two R9700s

The short path for a desktop or a small server with one or two AMD Radeon AI PRO R9700 cards: set it
up once, then start and stop the server with two commands. `README.md` has the full story and every
setting; this page is the day-to-day part.

| | One R9700 | Two R9700 |
|---|---|---|
| Engine | image `stilldeadcode/vllm-radiance:0.9.3`, single-GPU profile (`serve-tp1.sh`) | current vLLM with this repo's overlay (`vnext/serve.sh`, image in `vnext/BASE`) |
| Context | 220,000 tokens | 262,144 tokens |
| Requests at once | 3 | 4 (`MAXSEQS`) |
| One stream, short prompts | ~138 tok/s | ~145-158 tok/s |
| Prefill | ~2,500-2,800 tok/s | ~3,000-3,200 tok/s |
| Measured in | README.md, "One card (TP=1)" | vnext/RESULTS.md |

Both serve the same checkpoint with the same DFlash2 drafter, on the same port, under the same model
name, so a client does not notice which one is running.

## 1. Once: set up

You need Linux with the `amdgpu` driver, Docker (or Podman) able to reach `/dev/kfd` and `/dev/dri`,
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

`up.sh` counts the usable cards and picks the engine from the table above: two cards (or more, of
which it uses the first two) get the current-vLLM serve, one card the single-GPU profile.
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
