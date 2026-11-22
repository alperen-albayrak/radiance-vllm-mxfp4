"""Does a prefix-cache hit change the answer? Sends the same long prompt twice (cold, then cached)
and a third time with a new tail after the shared prefix; greedy. Prints answers and token counts.
Usage: python cachecorrect.py [url] [model] [prefix_tokens]"""
import json, random, sys, urllib.request
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
model = sys.argv[2] if len(sys.argv) > 2 else "test"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
r = random.Random(); W = "river stone cloud amber signal vector garden copper winter thread module buffer packet anchor".split()
facts = [f"Record {i}: the code for item {i} is {r.randrange(10**6):06d}." for i in range(n // 16)]
doc = f"[{r.getrandbits(40):x}]\n" + "\n".join(facts)
def ask(q):
    b = {"model": model, "messages": [{"role": "user", "content": doc + "\n\n" + q}], "max_tokens": 200, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    j = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    return j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"], (j["choices"][0]["message"].get("content") or "").strip()
k = len(facts) - 7; want = facts[k].split(" is ")[1].rstrip(".")
q1 = f"What is the code for item {k}? Reply with the code and one short sentence about where you found it."
for name, q in (("cold", q1), ("cached, same prompt", q1), ("cached prefix, new question", f"What is the code for item {k}? Reply with only the six digits.")):
    p, c, a = ask(q)
    print(f"{name:28} prompt={p} out={c:3} correct={want in a}  {a[:90]!r}")
