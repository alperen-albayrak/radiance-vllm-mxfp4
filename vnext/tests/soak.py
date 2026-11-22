"""Soak test for an OpenAI-compatible server: several clients send a random mix of requests for a
fixed time, and every reply that can be checked is checked.

Request kinds (picked at random by each worker):
  lookup     a document of numbered records (2K..60K tokens) and a question with one right answer.
             Documents are reused across workers with new questions, so the prefix cache is hit
             cold, warm and concurrently. Thinking on or off, greedy or sampled.
  grow       a conversation that grows by one turn per request (agent-like), answer checked each turn.
  tool       a request that must end in a tool call with the right argument.
  json       JSON-schema constrained output, parsed and checked.
  stream     a long streamed answer with thinking; one in three is aborted by closing the connection
             mid-stream (what a user pressing Esc in an agent does).
At the end: per kind counts, wrong answers, errors, latency, and whether the server still answers.

Usage: python soak.py <url> <model> <minutes> [workers=6] [seed]
"""
import json, random, sys, threading, time, urllib.request

url, model, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3])
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
seed = int(sys.argv[5]) if len(sys.argv) > 5 else int(time.time())
deadline = time.time() + minutes * 60
lock = threading.Lock()
stats = {}          # kind -> [n, wrong, errors, total seconds]
problems = []


def note(kind, ok, dt, detail=""):
    with lock:
        s = stats.setdefault(kind, [0, 0, 0, 0.0]); s[0] += 1; s[3] += dt
        if ok is False:
            s[1] += 1; problems.append(f"WRONG {kind}: {detail[:300]}")
        elif ok is None:
            s[2] += 1; problems.append(f"ERROR {kind}: {detail[:300]}")


def post(body, timeout=1800):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps({"model": model, **body}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


class Doc:
    def __init__(self, rng, tokens):
        self.codes = [f"{rng.randrange(10**6):06d}" for _ in range(max(8, tokens // 16))]
        self.text = f"[{rng.getrandbits(40):x}]\n" + "\n".join(f"Record {i}: the code for item {i} is {c}." for i, c in enumerate(self.codes))


docs = []


def get_doc(rng):
    with lock:
        if docs and rng.random() < 0.6:
            return rng.choice(docs)
        d = Doc(rng, rng.choice([2000, 4000, 8000, 16000, 30000, 60000]))
        docs.append(d)
        if len(docs) > 12:
            docs.pop(0)
        return d


def sampling(rng):
    return {"temperature": 0} if rng.random() < 0.5 else {"temperature": 0.7, "top_p": 0.95, "top_k": 20}


def k_lookup(rng):
    d = get_doc(rng); i = rng.randrange(len(d.codes)); think = rng.random() < 0.4
    r = post({"messages": [{"role": "user", "content": d.text + f"\n\nWhat is the code for item {i}? Reply with only the six digits."}],
              "max_tokens": 1500 if think else 60, **sampling(rng), "chat_template_kwargs": {"enable_thinking": think}})
    a = r["choices"][0]["message"].get("content") or ""
    return d.codes[i] in a, f"doc {len(d.codes)} records item {i} want {d.codes[i]} got {a!r} finish {r['choices'][0]['finish_reason']}"


def k_grow(rng):
    d = Doc(rng, rng.choice([3000, 12000])); msgs = [{"role": "user", "content": d.text + "\n\nI will ask about items one at a time. Reply with only the six digits each time. Ready?"}]
    r = post({"messages": msgs, "max_tokens": 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    msgs.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or "ok"})
    for _ in range(rng.randint(3, 6)):
        i = rng.randrange(len(d.codes)); msgs.append({"role": "user", "content": f"Item {i}?"})
        r = post({"messages": msgs, "max_tokens": 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        a = r["choices"][0]["message"].get("content") or ""; msgs.append({"role": "assistant", "content": a})
        if d.codes[i] not in a:
            return False, f"turn {len(msgs) // 2} item {i} want {d.codes[i]} got {a!r}"
    return True, ""


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def k_tool(rng):
    city = rng.choice(["Ankara", "Izmir", "Oslo", "Lima", "Kyoto", "Nairobi"])
    r = post({"messages": [{"role": "user", "content": f"What is the weather in {city} right now? Use the tool."}], "tools": TOOLS,
              "max_tokens": 600, **sampling(rng)})
    tc = (r["choices"][0]["message"].get("tool_calls") or [None])[0]
    ok = bool(tc) and tc["function"]["name"] == "get_weather" and city.lower() in json.loads(tc["function"]["arguments"]).get("city", "").lower()
    return ok, f"{city}: {tc and tc['function']} finish {r['choices'][0]['finish_reason']}"


def k_json(rng):
    name = rng.choice(["Ada", "Linus", "Grace", "Deniz"]); age = rng.randint(20, 80)
    r = post({"messages": [{"role": "user", "content": f"Give a JSON object for a person named {name} aged {age}."}], "max_tokens": 300, **sampling(rng),
              "response_format": {"type": "json_schema", "json_schema": {"name": "p", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}}}})
    c = r["choices"][0]["message"].get("content") or ""
    try:
        j = json.loads(c); return j.get("name") == name and j.get("age") == age, c
    except ValueError:
        return False, c


def k_stream(rng):
    topic = rng.choice(["how a B-tree stays balanced", "the causes of the First World War", "how TCP congestion control works",
                        "Rust ownership with examples", "PostgreSQL MVCC", "ekşi mayalı ekmek yapımı"])
    abort_after = rng.randint(20, 200) if rng.random() < 0.34 else None
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps({"model": model, "stream": True, "max_tokens": rng.choice([300, 800, 1500]),
          **sampling(rng), "messages": [{"role": "user", "content": f"Explain {topic} in detail."}]}).encode(), headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=1800); n = 0; text = ""; finish = None
    for line in resp:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        for ch in json.loads(line[6:]).get("choices", []):
            dl = ch.get("delta", {}); text += (dl.get("content") or "") + (dl.get("reasoning") or dl.get("reasoning_content") or ""); n += 1
            finish = ch.get("finish_reason") or finish
        if abort_after and n >= abort_after:
            resp.close(); return True, "aborted"
    return len(text) > 100 and finish in ("stop", "length"), f"len {len(text)} finish {finish}"


KINDS = [("lookup", k_lookup, 5), ("grow", k_grow, 2), ("tool", k_tool, 2), ("json", k_json, 1), ("stream", k_stream, 4)]


def worker(wid):
    rng = random.Random(seed * 100 + wid)
    while time.time() < deadline:
        kind, fn, _ = rng.choices(KINDS, weights=[k[2] for k in KINDS])[0]
        t = time.time()
        try:
            ok, detail = fn(rng); note(kind, ok, time.time() - t, detail)
        except Exception as e:  # noqa: BLE001
            body = getattr(e, "read", lambda: b"")()[:300]
            note(kind, None, time.time() - t, f"{type(e).__name__}: {e} {body!r}")
            time.sleep(2)


print(f"soak: {minutes} min, {workers} workers, seed {seed}", flush=True)
ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(workers)]
[t.start() for t in ts]
while any(t.is_alive() for t in ts):
    time.sleep(60)
    with lock:
        print(time.strftime("%H:%M"), {k: v[:3] for k, v in stats.items()}, flush=True)
alive = True
try:
    r = post({"messages": [{"role": "user", "content": "Reply with exactly: ok"}], "max_tokens": 20, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
    alive = "ok" in (r["choices"][0]["message"].get("content") or "").lower()
except Exception as e:  # noqa: BLE001
    alive = False; problems.append(f"server does not answer after the run: {e}")
print(f"{'kind':8} {'n':>5} {'wrong':>6} {'errors':>7} {'mean s':>7}")
for k, (n, w, e, s) in sorted(stats.items()):
    print(f"{k:8} {n:5d} {w:6d} {e:7d} {s / max(n, 1):7.1f}")
for p in problems[:40]:
    print(p)
bad = sum(v[1] + v[2] for v in stats.values())
print("RESULT:", "PASS" if alive and bad == 0 else f"{bad} problems, server {'answers' if alive else 'DOES NOT ANSWER'}")
