"""M1 reference for the gfx1201 port: HF transformers (PR #51050) on CPU, bf16.
Per prompt: input ids, the output of every decoder-layer visit (call order: prelude, 3 core
passes, coda = 48 visits), final-position logits (fp32), top-20, and a 64-token greedy continuation.
Usage: python ref_dump.py <model_dir> <out_dir>"""
import json, sys, time, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

model_dir, out_dir = sys.argv[1], sys.argv[2]
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu").eval()

LONG = ("The quick brown fox jumps over the lazy dog. " * 220).strip()
PROMPTS = {
    "capital": [{"role": "user", "content": "What is the capital of Turkey? One word."}],
    "turkish": [{"role": "user", "content": "Fotosentez ile hücresel solunum arasındaki farkı iki cümleyle açıkla."}],
    "code": [{"role": "user", "content": "Write a Rust function that returns the n-th Fibonacci number iteratively."}],
    "system": [{"role": "system", "content": "You are a terse assistant."}, {"role": "user", "content": "Name three prime numbers above 100."}],
    "multiturn": [{"role": "user", "content": "Pick a colour."}, {"role": "assistant", "content": "Blue."}, {"role": "user", "content": "Why that one?"}],
    "long": [{"role": "user", "content": LONG + "\n\nHow many times does the first sentence repeat? Estimate."}],
}

layers = model.model.language_model.layers
visits = []
def hook(idx):
    def f(mod, args, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        visits.append((idx, h.detach().float().clone()))
    return f
handles = [l.register_forward_hook(hook(i)) for i, l in enumerate(layers)]

meta = {"transformers": __import__("transformers").__version__, "torch": torch.__version__,
        "pr": "huggingface/transformers#51050", "dtype": "bfloat16", "activation_quant": False, "prompts": {}}
for name, msgs in PROMPTS.items():
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"]
    visits.clear(); t = time.time()
    with torch.no_grad():
        out = model(**enc)
    order = [i for i, _ in visits]
    # keep the last 8 positions of every visit (full tensors for the long prompt would be GBs)
    hs = torch.stack([h[0, -8:] for _, h in visits]).to(torch.float16)
    logits = out.logits[0, -1].float()
    top = torch.topk(logits, 20)
    handles_off = [h.remove() for h in handles]
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=64, do_sample=False)
    handles = [l.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
    new = gen[0, ids.shape[1]:]
    torch.save({"input_ids": ids[0], "visit_layers": order, "hidden_last8": hs, "logits_last": logits,
                "top20_ids": top.indices, "top20_logits": top.values, "greedy_ids": new}, f"{out_dir}/{name}.pt")
    meta["prompts"][name] = {"tokens": ids.shape[1], "visits": len(order), "visit_layers": order,
                             "top1": tok.decode(top.indices[:1]), "greedy": tok.decode(new), "seconds": round(time.time() - t)}
    print(name, ids.shape[1], "tok", len(order), "visits", repr(tok.decode(new)[:80]), flush=True)
json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1, ensure_ascii=False)
