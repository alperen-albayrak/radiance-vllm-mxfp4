"""Exercise RadianceQwenGatedDeltaNetAttention._forward_core routing with a fake radiance_gdn
(no GPU). usage: PYTHONPATH=<tree> python test_gdn_wrapper.py"""
import os, types
if not hasattr(os, "RTLD_DEEPBIND"):
    os.RTLD_DEEPBIND = 0
import radiance_v032_plugin as P
P.register_gdn_hook()
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as G
from vllm.model_executor.custom_op import op_registry_oot
Sub = op_registry_oot["QwenGatedDeltaNetAttention"]
log = []
G.QwenGatedDeltaNetAttention._forward_core = lambda self, *a: log.append("stock")
fc = types.SimpleNamespace(attn_metadata=None)
G.get_forward_context = lambda: fc
def mk(fused_ret):
    fake = types.SimpleNamespace(ALL=True, forward_core_fused=lambda *a: log.append("fused") or fused_ret)
    inst = object.__new__(Sub)
    inst.__dict__.update(prefix="L0", _radiance_gdn=fake)
    return inst
def run(inst, md):
    log.clear(); fc.attn_metadata = md; inst._forward_core(1, 2, 3, 4); return list(log)
assert run(mk(True), {"L0": object()}) == ["fused"], log            # R4D takes the step
assert run(mk(False), {"L0": object()}) == ["fused", "stock"], log  # R4D declines -> stock
assert run(mk(True), None) == ["stock"], log                        # warm-up: stock only
assert run(mk(True), {"other": 1}) == ["stock"], log                # no metadata for layer
off = object.__new__(Sub); off.__dict__.update(prefix="L0", _radiance_gdn=None)
assert run(off, {"L0": object()}) == ["stock"], log                 # R4D off -> stock
print("GDN wrapper routing: 5/5 OK")
# __init__ steering: stub the heavy stock __init__ with the two attributes it would set on ROCm
import sys, torch
def stub_init(self, *a, **k):
    torch.nn.Module.__init__(self)
    self.enable_fused_gdn_spec_decode = True
    self._forward_method = self.forward_hip
G.QwenGatedDeltaNetAttention.__init__ = stub_init
sys.modules["radiance_gdn"] = types.SimpleNamespace(ALL=True)
x = G.QwenGatedDeltaNetAttention()          # in-tree class -> OOT subclass
assert type(x) is Sub
assert x.enable_fused_gdn_spec_decode is False
assert x._forward_method.__func__ is G.QwenGatedDeltaNetAttention.forward_cuda
sys.modules["radiance_gdn"] = types.SimpleNamespace(ALL=False)
y = G.QwenGatedDeltaNetAttention()
assert y.enable_fused_gdn_spec_decode is True
assert y._forward_method.__func__ is G.QwenGatedDeltaNetAttention.forward_hip
print("GDN __init__ steering (R4D on/off): 2/2 OK")
