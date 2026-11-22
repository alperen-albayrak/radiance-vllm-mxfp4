"""Four measurements on a running server (see TODO "Ideas from other engines").
  m.py hit   <url> <model>            draft acceptance: cold prompt vs the same prompt from the prefix cache
  m.py stall <url> <model> [tokens]   decode speed of two sessions before/while a third prefills a long cold prompt
  m.py steps <url> <model> <ctx,...>  ms per speculative step at the given context sizes (cached context)
"""
import json, os, random, re, sys, threading, time, urllib.request
mode, url, model = sys.argv[1:4]
# .py files the long prompts are cut from: this repository unless CORPUS names another tree
CORPUS = os.environ.get("CORPUS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")

def post(path, body, stream=False, timeout=3600):
    r = urllib.request.urlopen(urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=timeout)
    return r if stream else json.load(r)

def counters():
    t = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode(); o = {}
    for k in ("num_drafts", "num_accepted_tokens"):
        o[k] = sum(float(x) for x in re.findall(rf"^vllm:spec_decode_{k}(?:_total)?(?:{{[^}}]*}})? ([0-9.e+]+)$", t, re.M))
    return o

_ids = None
def corpus_ids(n):
    global _ids
    if _ids is None or len(_ids) < n:
        files = []
        for root, _, names in os.walk(CORPUS):
            files += [os.path.join(root, f) for f in names if f.endswith(".py")]
        random.Random(1).shuffle(files); _ids = []; buf = ""
        for f in files:
            buf += open(f, errors="ignore").read() + "\n\n"
            if len(buf) > 400_000:
                _ids += post("/tokenize", {"model": model, "prompt": buf, "add_special_tokens": False})["tokens"]; buf = ""
                if len(_ids) >= n: break
    return _ids

def text(start, n):
    return post("/detokenize", {"model": model, "tokens": corpus_ids(start + n)[start:start + n]})["prompt"]

def gen(prompt, max_tokens, on_token=None, **kw):
    """Streamed chat request. Returns dict(ttft, dur, completion, cached, prompt, drafts, accepted)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0, **kw}
    c0 = counters(); t0 = time.time(); first = None; usage = {}
    for line in post("/v1/chat/completions", body, stream=True):
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
        d = json.loads(line[6:]); usage = d.get("usage") or usage
        for ch in d.get("choices", []):
            dl = ch.get("delta", {})
            if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                if first is None: first = time.time()
                if on_token: on_token(time.time())
    end = time.time(); c1 = counters()
    return {"ttft": (first or end) - t0, "dur": end - (first or end), "completion": usage.get("completion_tokens", 0), "prompt": usage.get("prompt_tokens", 0),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "drafts": c1["num_drafts"] - c0["num_drafts"], "accepted": c1["num_accepted_tokens"] - c0["num_accepted_tokens"]}

Q = "\n\nWrite a detailed, section-by-section explanation of what the code above does. Do not stop early."
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}

if mode == "hit":
    # Greedy, thinking off: the cold and the cached request produce the same text, so their
    # acceptance is directly comparable. "new tail": cached prefix, other question (short new suffix).
    rows = []
    for i, size in enumerate([6000, 6000, 20000, 20000, 60000, 60000]):
        doc = f"[{random.getrandbits(48):012x}]\n" + text(i * 61000, size)
        a = gen(doc + Q, 400, **NOTHINK); b = gen(doc + Q, 400, **NOTHINK)
        c = gen(doc + "\n\nList every function defined above with a one-line description of each.", 400, **NOTHINK)
        d = gen(f"[{random.getrandbits(48):012x}]\n" + doc.split("\n", 1)[1] + "\n\nList every function defined above with a one-line description of each.", 400, **NOTHINK)
        al = lambda r: 1 + r["accepted"] / max(r["drafts"], 1)
        rows.append((size, al(a), al(b), al(c), al(d), b["cached"], c["cached"], a["completion"], b["completion"]))
        print(f"ctx {size:6d}: acceptance cold {al(a):.2f} | same prompt cached {al(b):.2f} (cached {b['cached']}) | cached prefix, other question {al(c):.2f} (cached {c['cached']}) | that question cold {al(d):.2f}", flush=True)
    m = lambda j: sum(r[j] for r in rows) / len(rows)
    print(f"mean: cold {m(1):.2f} cached-same {m(2):.2f} | other question: cached {m(3):.2f} cold {m(4):.2f}")

elif mode == "stall":
    big = int(sys.argv[4]) if len(sys.argv) > 4 else 250000
    stamps = [[], []]; done = threading.Event()
    def decoder(k):
        doc = f"[{random.getrandbits(48):012x}]\n" + text(k * 9000, 8000)
        while not done.is_set():
            gen(doc + Q, 3000, on_token=stamps[k].append)
    ts = [threading.Thread(target=decoder, args=(k,), daemon=True) for k in range(2)]
    [t.start() for t in ts]
    time.sleep(40); t1 = time.time()                       # both decoding from their caches by now
    time.sleep(30); t2 = time.time()
    res = {}
    def prefill():
        res["r"] = gen(f"[{random.getrandbits(48):012x}]\n" + text(300000, big - 200) + Q, 1)
    p = threading.Thread(target=prefill); p.start(); p.join(); t3 = time.time(); done.set()
    def rate(k, a, b):   # stream chunks per second (a chunk carries one or more tokens; comparable within a run)
        return sum(1 for x in stamps[k] if a <= x < b) / (b - a)
    def gaps(k, a, b):
        xs = [x for x in stamps[k] if a <= x < b]; g = sorted(y - x for x, y in zip(xs, xs[1:]))
        return (g[len(g) // 2], g[int(len(g) * 0.99)], g[-1]) if g else (0, 0, 0)
    r = res["r"]
    print(f"cold prefill of {r['prompt']} tokens alongside 2 decoding sessions: first token after {r['ttft']:.1f} s = {r['prompt'] / r['ttft']:.0f} tok/s")
    for k in range(2):
        g0, g1 = gaps(k, t1, t2), gaps(k, t2, t3)
        print(f"session {k}: chunks/s alone-ish {rate(k, t1, t2):.1f} -> during prefill {rate(k, t2, t3):.1f} | gap between chunks median/p99/max: {g0[0]*1000:.0f}/{g0[1]*1000:.0f}/{g0[2]*1000:.0f} ms -> {g1[0]*1000:.0f}/{g1[1]*1000:.0f}/{g1[2]*1000:.0f} ms")

elif mode == "steps":
    for size in [int(x) for x in sys.argv[4].split(",")]:
        doc = f"[{random.getrandbits(48):012x}]\n" + text(0, size - 150)
        gen(doc + Q, 1, **NOTHINK)                          # prefill
        r = gen(doc + Q, 500, **NOTHINK)                    # decode from the cache
        steps = r["drafts"] or r["completion"]
        print(f"ctx {size:6d}: {r['completion']} tokens in {r['dur']:.2f} s = {r['completion'] / r['dur']:.1f} tok/s | {steps:.0f} steps, {r['dur'] / steps * 1000:.1f} ms/step | acceptance length {1 + r['accepted'] / max(r['drafts'], 1):.2f} | cached {r['cached']}", flush=True)

if mode == "evict":
    # Four cold prompts of `big` tokens overflow the GPU pool; then the first one is asked again.
    big = int(sys.argv[4]) if len(sys.argv) > 4 else 250000
    docs = [f"[{random.getrandbits(48):012x}]\n" + text(k * (big + 1000), big - 200) + Q for k in range(4)]
    for k, d in enumerate(docs):
        r = gen(d, 1, **NOTHINK); print(f"prompt {k}: {r['prompt']} tokens, first token after {r['ttft']:.1f} s, cached {r['cached']}", flush=True)
    for k in (0, 3, 1):
        r = gen(docs[k], 200, **NOTHINK)
        print(f"again prompt {k}: first token after {r['ttft']:.1f} s, cached {r['cached']} of {r['prompt']}, then {r['completion'] / max(r['dur'], 1e-6):.0f} tok/s", flush=True)

if mode == "evictcheck":
    # As "evict", with record documents whose answers can be checked, so a wrong restored state shows.
    big = int(sys.argv[4]) if len(sys.argv) > 4 else 240000
    rng = random.Random()
    def mk():
        codes = [f"{rng.randrange(10**6):06d}" for _ in range(big // 16)]
        return codes, f"[{rng.getrandbits(40):x}]\n" + "\n".join(f"Record {i}: the code for item {i} is {c}." for i, c in enumerate(codes))
    docs = [mk() for _ in range(4)]
    def ask(k, tag):
        codes, doc = docs[k]; i = rng.randrange(len(codes) - 2000, len(codes)) if rng.random() < 0.5 else rng.randrange(len(codes))
        body = {"model": model, "messages": [{"role": "user", "content": doc + f"\n\nWhat is the code for item {i}? Reply with only the six digits."}],
                "max_tokens": 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time(); r = post("/v1/chat/completions", body); a = r["choices"][0]["message"].get("content") or ""
        print(f"{tag} doc {k} item {i}: {time.time() - t:6.1f} s  correct={codes[i] in a}  prompt {r['usage']['prompt_tokens']}  answer {a.strip()[:20]!r}", flush=True)
    for k in range(4): ask(k, "cold ")
    for k in (0, 1, 2, 3, 0, 2): ask(k, "again")
