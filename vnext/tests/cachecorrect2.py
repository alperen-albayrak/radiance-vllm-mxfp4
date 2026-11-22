"""Second cache-hit check: the first request generates only `first_max` tokens (as a client that
only wants the prompt cached, or a 1-token probe), the second asks the same prompt for a full answer.
Usage: python cachecorrect2.py <url> <model> <tokens> <first_max> <thinking 0|1> <temp>"""
import json, random, sys, urllib.request
url, model, n, first_max, think, temp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1", float(sys.argv[6])
r = random.Random()
facts = [f"Record {i}: the code for item {i} is {r.randrange(10**6):06d}." for i in range(n // 16)]
k = len(facts) - 7; want = facts[k].split(" is ")[1].rstrip(".")
prompt = f"[{r.getrandbits(40):x}]\n" + "\n".join(facts) + f"\n\nWhat is the code for item {k}? Reply with the code and one short sentence."
def ask(mt):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": temp,
         "chat_template_kwargs": {"enable_thinking": think}}
    j = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    m = j["choices"][0]["message"]; u = j["usage"]
    return u["completion_tokens"], ((u.get("prompt_tokens_details") or {}).get("cached_tokens")), j["choices"][0]["finish_reason"], (m.get("reasoning") or m.get("reasoning_content") or ""), (m.get("content") or "")
c1 = ask(first_max); c2 = ask(400)
print(f"first_max={first_max} think={int(think)} temp={temp}: first out={c1[0]} | second out={c2[0]} cached={c2[1]} finish={c2[2]} correct={want in c2[4]} reasoning={c2[3][:50]!r} answer={c2[4][:60]!r}")
