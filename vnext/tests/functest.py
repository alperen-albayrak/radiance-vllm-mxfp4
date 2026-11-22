"""Functional checks against an OpenAI-compatible server: tool call, tool result round trip, image
input, streaming with reasoning, JSON schema output. Usage: python functest.py <url> <model>"""
import base64, io, json, struct, sys, urllib.request, zlib
url, model = sys.argv[1], sys.argv[2]

def post(body):
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps({"model": model, **body}).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))

def png(w, h, rgb):  # solid-colour PNG without external libraries
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, str(detail)[:160])

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
m1 = [{"role": "user", "content": "What is the weather in Ankara right now, in celsius? Use the tool."}]
r = post({"messages": m1, "tools": tools, "max_tokens": 400, "temperature": 0})
msg = r["choices"][0]["message"]; tc = (msg.get("tool_calls") or [None])[0]
args = json.loads(tc["function"]["arguments"]) if tc else {}
check("tool call emitted", tc and tc["function"]["name"] == "get_weather" and "ankara" in args.get("city", "").lower(), tc and tc["function"])
check("finish_reason tool_calls", r["choices"][0]["finish_reason"] == "tool_calls", r["choices"][0]["finish_reason"])
if tc:
    m2 = m1 + [{"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg["tool_calls"]},
               {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"city": "Ankara", "temp_c": 7, "sky": "overcast"})}]
    r2 = post({"messages": m2, "tools": tools, "max_tokens": 400, "temperature": 0})
    c2 = r2["choices"][0]["message"].get("content") or ""
    check("tool result used", "7" in c2 and r2["choices"][0]["finish_reason"] == "stop", c2.strip()[:120])

img = base64.b64encode(png(64, 64, (220, 20, 20))).decode()
r = post({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
          {"type": "text", "text": "What single colour fills this image? One word."}]}], "max_tokens": 300, "temperature": 0})
c = (r["choices"][0]["message"].get("content") or "").lower()
check("image input", "red" in c, c.strip()[:80])

r = post({"messages": [{"role": "user", "content": "Give a JSON object for a person named Ada aged 36."}], "max_tokens": 300, "temperature": 0,
          "response_format": {"type": "json_schema", "json_schema": {"name": "p", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}}}})
try:
    j = json.loads(r["choices"][0]["message"]["content"]); check("json schema output", j.get("name") == "Ada" and j.get("age") == 36, j)
except Exception as e:
    check("json schema output", False, repr(e) + " " + repr(r["choices"][0]["message"].get("content"))[:100])

req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps({"model": model, "stream": True, "max_tokens": 300, "temperature": 0,
      "messages": [{"role": "user", "content": "What is 12*12? Answer briefly."}]}).encode(), headers={"Content-Type": "application/json"})
reason = content = ""
for line in urllib.request.urlopen(req, timeout=600):
    if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
        for ch in json.loads(line[6:]).get("choices", []):
            d = ch.get("delta", {}); reason += d.get("reasoning") or d.get("reasoning_content") or ""; content += d.get("content") or ""
check("streaming: reasoning separated, answer right", len(reason) > 0 and "144" in content and "</think>" not in content, content.strip()[:60])

r = post({"messages": [{"role": "user", "content": "Reply with exactly: ok"}], "max_tokens": 50, "temperature": 0,
          "chat_template_kwargs": {"enable_thinking": False}})
m = r["choices"][0]["message"]; check("thinking off", not (m.get("reasoning") or m.get("reasoning_content")) and "ok" in (m.get("content") or "").lower(), m.get("content"))
print("ALL PASS" if ok else "SOME FAILED")
