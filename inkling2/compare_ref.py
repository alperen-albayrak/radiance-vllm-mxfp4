"""Compare a running server with the M1 reference dumps (ref_dump.py).
For every prompt: greedy continuation from the reference's input ids (token ids sent as the
prompt), agreement of the first token, length of the common greedy prefix, and the reference
top-1's rank/logprob in the server's first-step top-20.
Usage: python compare_ref.py <ref_dir> [url] [model]"""
import json, sys, time, urllib.request
import torch

ref_dir = sys.argv[1]
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8090"
model = sys.argv[3] if len(sys.argv) > 3 else "inkling-2-mini"
meta = json.load(open(f"{ref_dir}/meta.json"))
total = agree = 0
for name in meta["prompts"]:
    d = torch.load(f"{ref_dir}/{name}.pt")
    ids = d["input_ids"].tolist(); ref = d["greedy_ids"].tolist()
    body = {"model": model, "prompt": ids, "max_tokens": len(ref), "temperature": 0, "logprobs": 20,
            "return_tokens_as_token_ids": True, "skip_special_tokens": False, "add_special_tokens": False}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                                                 headers={"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t
    lp = r["choices"][0]["logprobs"]
    got = [int(t.split(":")[1]) for t in lp["tokens"]]
    prefix = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
    first = {int(k.split(":")[1]): v for k, v in lp["top_logprobs"][0].items()}
    ref_top = d["top20_ids"].tolist()
    overlap = len(set(ref_top[:5]) & set(sorted(first, key=first.get, reverse=True)[:5]))
    same = sum(a == b for a, b in zip(got, ref))
    total += len(ref); agree += same
    print(f"{name:10} {len(ids):5} prompt tok | first token {'same' if got[:1] == ref[:1] else 'DIFF'} | common greedy prefix {prefix}/{len(ref)} | "
          f"same positions {same}/{len(ref)} | top-5 overlap at step 0: {overlap}/5 | {len(got) / dt:.1f} tok/s")
print(f"token agreement over all prompts: {agree}/{total} = {agree / total:.1%}")
