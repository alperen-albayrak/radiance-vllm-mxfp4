"""Run the upstream ROCm (gfx950) Inkling2 ops unchanged on this GPU and compare with torch.
Loads the ops from an integration tree (argv[1]) without importing vllm.models.inkling2."""
import importlib.util, inspect, sys, types, traceback
import torch

tree = sys.argv[1]
base = f"{tree}/vllm/models/inkling2/amd/ops/"
pkg = types.ModuleType("i2ops"); pkg.__path__ = [base]; sys.modules["i2ops"] = pkg
def load(name):
    spec = importlib.util.spec_from_file_location("i2ops." + name, base + name + ".py")
    m = importlib.util.module_from_spec(spec); sys.modules["i2ops." + name] = m; spec.loader.exec_module(m); return m

dev = "cuda"; torch.manual_seed(0)
print("device", torch.cuda.get_device_name(0), "torch", torch.__version__)
results = {}
def check(name, fn):
    try:
        err = fn(); results[name] = f"ok  max rel err {err:.2e}"
    except Exception as e:
        results[name] = "FAIL " + type(e).__name__ + ": " + str(e).split("\n")[0][:170]
    print(f"{name:28} {results[name]}", flush=True)

norm = load("norm")
def t_rms():
    x = torch.randn(37, 2304, device=dev).bfloat16() * 3; w = torch.randn(2304, device=dev).bfloat16()
    ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * w.float()
    out = norm.rms_norm(x, w, 1e-6, torch.bfloat16)
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()
check("rms_norm", t_rms)

linear = load("linear")
print("linear_k_groups sig:", inspect.signature(linear.linear_k_groups))
def t_linear():
    x = torch.randn(33, 4608, device=dev).bfloat16(); w = torch.randn(64, 4608, device=dev).bfloat16() * 0.02
    ref = x.float() @ w.float().t()
    out = linear.linear_k_groups(x, w, torch.bfloat16)
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()
check("linear_k_groups", t_linear)

attn_res = load("attn_res")
print("attn_res sig:", inspect.signature(attn_res.attn_res))
def t_attn_res():
    T, D, B = 19, 2304, 3
    stack = torch.randn(B, T, D, device=dev).bfloat16(); last = torch.randn(T, D, device=dev).bfloat16()
    q = torch.randn(1, D, device=dev) * D ** -0.5; w = torch.ones(1, D, device=dev).bfloat16()
    out = attn_res.attn_res(stack, last, None, q, w, 1e-6, True)
    assert torch.isfinite(out.float()).all()
    return 0.0
check("attn_res (runs, finite)", t_attn_res)

mxfp8 = load("mxfp8")
print("mxfp8_quantize sig:", inspect.signature(mxfp8.mxfp8_quantize))
bmm = load("triton_bmm")
print("bmm sig:", inspect.signature(bmm.bmm), "| expert_plan sig:", inspect.signature(bmm.expert_plan))
def t_mxfp8():
    x = torch.randn(16, 2304, device=dev).bfloat16()
    out = mxfp8.mxfp8_quantize(x)
    return 0.0
check("mxfp8_quantize (runs)", t_mxfp8)
moe = load("moe")
for n in ("route", "combine"): print(n, "sig:", inspect.signature(getattr(moe, n)))

# --- the gfx950 MXFP8 expert GEMM (tl.dot_scaled) on this GPU ---
def t_bmm():
    E, N, K = 8, 2304, 2304
    plan = bmm.ExpertPlan(tile=16, max_ctas=2)
    act = (torch.randn(plan.rows, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    w = (torch.randn(E, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    # scale exponent 127 = 2**0 everywhere, so the layout of the scale tiles does not matter
    a_s = torch.full((plan.rows * (K // 32) + 4096,), 127, dtype=torch.uint8, device=dev)
    w_s = torch.full((E, N, K // 32), 127, dtype=torch.uint8, device=dev)
    tiles = torch.tensor([3, 5, 16, 28, 2, 0, 0], dtype=torch.int32, device=dev)
    out = bmm.bmm(act, a_s, w, w_s, tiles, plan)
    ref0 = act[:16].float() @ w[3].float().t(); ref1 = act[16:28].float() @ w[5].float().t()
    e0 = (out[:16].float() - ref0).abs().max() / ref0.abs().max()
    e1 = (out[16:28].float() - ref1).abs().max() / ref1.abs().max()
    return max(e0.item(), e1.item())
check("mxfp8 bmm (dot_scaled)", t_bmm)
