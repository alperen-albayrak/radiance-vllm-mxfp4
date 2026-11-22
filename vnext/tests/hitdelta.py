"""Exact prefix-cache reuse per turn from the server's own counters (one session, nothing else running).
Usage: python hitdelta.py [url] [model]"""
import json, random, re, sys, time, urllib.request
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
model = sys.argv[2] if len(sys.argv) > 2 else "DaisyRose-27B"
W = "river stone cloud amber signal vector garden copper winter thread module buffer packet anchor".split()
r = random.Random()
def text(n): return " ".join(r.choice(W) for _ in range(n))
def counters():
    t = urllib.request.urlopen(url + "/metrics").read().decode()
    g = lambda k: sum(float(x) for x in re.findall(rf"^vllm:prefix_cache_{k}_total(?:{{[^}}]*}})? ([0-9.e+]+)$", t, re.M))
    return g("queries"), g("hits")
def ask(msgs):
    b = {"model": model, "messages": msgs, "max_tokens": 80}
    t0 = time.time()
    j = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    return j["usage"]["prompt_tokens"], j["choices"][0]["message"].get("content") or "ok", time.time() - t0
msgs = [{"role": "system", "content": f"session {r.getrandbits(48):x}. Notes: " + text(8000)}, {"role": "user", "content": "Reply with one short sentence."}]
print(f"{'turn':>4} {'prompt':>7} {'new':>6} {'hit':>7} {'prefilled':>9} {'lost':>6}")
prev = 0
for i, add in enumerate([0, 300, 300, 300, 300, 1000, 300, 300, 2000, 300, 300]):
    if i: msgs.append({"role": "user", "content": "Tool result: " + text(add) + "\nReply with one short sentence."})
    q0, h0 = counters(); p, c, dt = ask(msgs); q1, h1 = counters()
    hit = int(h1 - h0); pre = int(q1 - q0) - hit
    print(f"{i:>4} {p:>7} {p - prev:>6} {hit:>7} {pre:>9} {max(prev - hit, 0):>6}", flush=True)
    msgs.append({"role": "assistant", "content": c}); prev = p
