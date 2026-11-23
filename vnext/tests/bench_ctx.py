"""Prefill / decode benchmark by context length and parallel sessions for an OpenAI-compatible
server with prefix caching (written for the radiance vLLM serves on 2 x R9700).

Every (context tokens, sessions) cell runs in two phases, so that prefill and decode do not
disturb each other:
  1. prefill: `sessions` chat requests start at the same instant, max_tokens=1. Each prompt is a
     different slice of a text corpus (the prefix cache never hits) plus a request for a long
     explanation.   prefill tok/s = prompt_tokens / time to first token   (per session, mean)
     prefill agg   = sum(prompt_tokens) / time until the last session's first token
  2. decode: the same prompts again, at the same instant, max_tokens=MAX_TOKENS. They come from
     the prefix cache (cached_tokens is recorded), so all sessions decode together at that
     context.        decode tok/s = (completion_tokens - 1) / (end - first token)   (mean)
     decode agg    = sum(completion_tokens) / (last end - first first-token)
If the contexts do not fit the KV cache together, phase 2 has to prefill again: `cached_frac`
shows it and the decode numbers of that cell then include prefill interference.
Sampling uses the server's defaults.

Usage: python bench_ctx.py <model> <out.json> [url] [corpus_dir]
  corpus_dir: .py files to cut the prompts from; default this repository (~2.5M tokens). Numbers
  in RESULTS.md were taken with a vLLM source tree, so compare runs made with the same corpus.
  SIZES, SESSIONS, MAX_TOKENS in the environment change the grid.
"""
import json, os, random, sys, threading, time, urllib.request

model, out_path = sys.argv[1], sys.argv[2]
url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080"
corpus_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SIZES = [int(x) for x in os.environ.get("SIZES", "1000,8000,32000,128000,250000").split(",")]
SESSIONS = [int(x) for x in os.environ.get("SESSIONS", "1,2,3,4").split(",")]
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "600"))
OVERHEAD = 120          # chat template + instruction, measured below and corrected per request
QUESTION = ("\n\nWrite a long, detailed, section-by-section explanation of what the code above does. "
            "Do not stop early.")


def post(path, body, timeout=3600, stream=False):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp if stream else json.load(resp)


def build_corpus(need_tokens):
    files = []
    for root, _, names in os.walk(corpus_dir):
        files += [os.path.join(root, n) for n in names if n.endswith(".py")]
    random.Random(0).shuffle(files)
    ids, buf = [], ""
    for f in files:
        buf += open(f, errors="ignore").read() + "\n\n"
        if len(buf) > 400_000:
            ids += post("/tokenize", {"model": model, "prompt": buf, "add_special_tokens": False})["tokens"]
            buf = ""
            if len(ids) >= need_tokens:
                break
    return ids


def one(prompt_text, res, i, start_evt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt_text}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    start_evt.wait()
    t0 = time.time(); first = None; usage = None; err = None
    try:
        for line in post("/v1/chat/completions", body, stream=True):
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                    first = time.time()
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e} {getattr(e, 'read', lambda: b'')()[:300]!r}"
    res[i] = {"t0": t0, "first": first, "end": time.time(), "usage": usage, "error": err}


def detok(ids):
    return post("/detokenize", {"model": model, "tokens": ids})["prompt"]


def main():
    need = max(SIZES) * max(SESSIONS) + 300_000
    print(f"tokenizing corpus (~{need} tokens)…", flush=True)
    corpus = build_corpus(need)
    print("corpus tokens:", len(corpus), flush=True)
    results = {"model": model, "url": url, "max_tokens": MAX_TOKENS, "cells": []}
    cursor = 0
    for size in SIZES:
        for n in SESSIONS:
            prompts = []
            for _ in range(n):
                take = size - OVERHEAD
                if cursor + take > len(corpus):
                    cursor = random.randint(1, 5000)
                prompts.append(f"[{random.getrandbits(48):012x}]\n" + detok(corpus[cursor:cursor + take]) + QUESTION)
                cursor += take
            def run(max_tokens):
                res = [None] * n; evt = threading.Event()
                threads = [threading.Thread(target=one, args=(p, res, i, evt, max_tokens)) for i, p in enumerate(prompts)]
                [t.start() for t in threads]; start = time.time(); evt.set(); [t.join() for t in threads]
                ok = [r for r in res if r["error"] is None and r["usage"] and r["first"]]
                return res, ok, start

            cell = {"context": size, "sessions": n}
            res, ok, start = run(1)
            cell["prefill_ok"] = len(ok); cell["errors"] = [r["error"] for r in res if r["error"]][:2]
            if ok:
                pt = [r["usage"]["prompt_tokens"] for r in ok]
                cell.update({
                    "prompt_tokens": round(sum(pt) / len(pt)),
                    "ttft_s": round(sum(r["first"] - r["t0"] for r in ok) / len(ok), 2),
                    "prefill_tok_s": round(sum(p / (r["first"] - r["t0"]) for p, r in zip(pt, ok)) / len(ok)),
                    "prefill_agg_tok_s": round(sum(pt) / (max(r["first"] for r in ok) - start)),
                    "prefill_wall_s": round(max(r["first"] for r in ok) - start, 1),
                })
            res, ok, start = run(MAX_TOKENS)
            cell["decode_ok"] = len(ok); cell["errors"] += [r["error"] for r in res if r["error"]][:2]
            if ok:
                pt = [r["usage"]["prompt_tokens"] for r in ok]; ct = [r["usage"]["completion_tokens"] for r in ok]
                cached = [((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0) for r in ok]
                cell.update({
                    "completion_tokens": round(sum(ct) / len(ct)),
                    "cached_frac": round(sum(cached) / max(sum(pt), 1), 3),
                    "decode_ttft_s": round(sum(r["first"] - r["t0"] for r in ok) / len(ok), 2),
                    "decode_tok_s": round(sum((c - 1) / max(r["end"] - r["first"], 1e-6) for c, r in zip(ct, ok)) / len(ok), 1),
                    "decode_agg_tok_s": round(sum(ct) / max(max(r["end"] for r in ok) - min(r["first"] for r in ok), 1e-6), 1),
                })
            results["cells"].append(cell)
            print(json.dumps(cell), flush=True)
            json.dump(results, open(out_path, "w"), indent=1)


main()
