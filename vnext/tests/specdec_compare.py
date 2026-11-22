"""Acceptance length and speed of one speculative-decoding setup on a running vLLM server.

Runs a fixed prompt set one request at a time (concurrency 1), in two sampling modes
(T=0 greedy; T=1.0 / top_p 0.95 / top_k 20 = the model's recommended thinking-mode sampling),
and reads the server's spec-decode counters before and after:
    acceptance length = 1 + accepted draft tokens / drafts        (tokens per verification step)
    tok/s             = completion tokens / wall time of the run

Usage: python specdec_compare.py <label> <model> <out.json> [url]
"""
import json, re, sys, time, urllib.request

label, model, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
url = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8090"
MAX_TOKENS = 512

PROMPTS = {
    "math": [
        "A train leaves at 9:40 and travels 150 km at 60 km/h, waits 25 minutes, then travels 90 km at 45 km/h. At what time does it arrive? Show the steps.",
        "Tom has 3 times as many marbles as Ann. After Tom gives Ann 12 marbles they have the same number. How many did each start with?",
        "A shop sells pens at 3 for $2 and notebooks at $4 each. Maria buys 12 pens and 5 notebooks and pays with a $50 bill. How much change does she get?",
        "Find the sum of all positive integers n < 100 such that n^2 + 1 is divisible by 5.",
    ],
    "code": [
        "Write a Python function that merges overlapping intervals in a list of (start, end) tuples and returns the merged list sorted by start. Include a few asserts.",
        "Write a Rust function that parses a semantic version string like '1.2.3-beta.1' into a struct and implements ordering for it.",
        "Implement an LRU cache in TypeScript with get and put in O(1), with a short usage example.",
        "Write a SQL query that returns, for each customer, their second most recent order date, given tables customers(id, name) and orders(id, customer_id, created_at).",
    ],
    "chat": [
        "Explain the difference between a process and a thread to someone who has only used spreadsheets.",
        "Give me a one-week beginner plan for learning to bake sourdough bread.",
        "What are the trade-offs between SQLite and PostgreSQL for a small self-hosted web app?",
        "Summarize the causes of the fall of the Western Roman Empire in about 200 words.",
    ],
    "turkish": [
        "Bir şehirde toplu taşıma ile özel araç kullanımının avantajlarını, dezavantajlarını ve hangisinin ne zaman tercih edileceğini açıkla.",
        "Rust dilinde sahiplik (ownership) ve ödünç alma (borrowing) kavramlarını kısa örneklerle anlat.",
        "İstanbul'dan Kapadokya'ya üç günlük bir gezi planı hazırla.",
    ],
    "agent": [
        "You are working in a git repository. The test `test_parse_date` fails with `ValueError: time data '2024-13-01' does not match format '%Y-%m-%d'`. Describe step by step which shell commands you would run to locate the bug and write the patch you expect to make.",
        "Plan the migration of a cron-based nightly backup script to a systemd timer on Debian 12. List every file you would create with its full content.",
        "A user reports that `docker compose up` fails with 'port is already allocated' for port 5432. Walk through the diagnosis with the exact commands and their expected output, then give the fix.",
    ],
}
MODES = {"T0": {"temperature": 0}, "T1": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}}


def counters():
    txt = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    out = {}
    for key in ("num_drafts", "num_accepted_tokens", "num_draft_tokens"):
        m = re.findall(rf"^vllm:spec_decode_{key}(?:_total)?(?:{{[^}}]*}})? ([0-9.e+]+)$", txt, re.M)
        out[key] = sum(float(x) for x in m)
    return out


def ask(prompt, sampling):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS, **sampling}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r["usage"]["completion_tokens"], time.time() - t


result = {"label": label, "model": model, "max_tokens": MAX_TOKENS, "runs": {}}
ask("Say hello.", {"temperature": 0})  # warm-up, not counted
for mode, sampling in MODES.items():
    for cat, prompts in PROMPTS.items():
        c0 = counters(); toks = 0; wall = 0.0
        for p in prompts:
            n, dt = ask(p, sampling); toks += n; wall += dt
        c1 = counters()
        drafts = c1["num_drafts"] - c0["num_drafts"]; acc = c1["num_accepted_tokens"] - c0["num_accepted_tokens"]
        run = {"tokens": toks, "tok_s": round(toks / wall, 1),
               "acceptance_length": round(1 + acc / drafts, 2) if drafts else None}
        result["runs"][f"{mode}/{cat}"] = run
        print(label, mode, cat, run, flush=True)
    for mode_ in (mode,):
        cells = [v for k, v in result["runs"].items() if k.startswith(mode_ + "/")]
        tot = sum(c["tokens"] for c in cells)
        result[f"{mode_}_mean_tok_s"] = round(tot / sum(c["tokens"] / c["tok_s"] for c in cells), 1)
        al = [c["acceptance_length"] for c in cells if c["acceptance_length"]]
        result[f"{mode_}_mean_acceptance"] = round(sum(al) / len(al), 2) if al else None
json.dump(result, open(out_path, "w"), indent=1)
print(json.dumps({k: v for k, v in result.items() if k != "runs"}))
