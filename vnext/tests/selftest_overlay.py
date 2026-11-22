"""Local (CPU/macOS, no GPU) checks of the Radiance v0.32 overlay against a vLLM source tree.

usage: PYTHONPATH=<tree> python test_overlay_local.py <tree>
<tree> = vLLM v0.32.0 checkout with the overlay applied (radiance_*.py + radiance_v032_plugin
at its root, next to vllm/). Prints PASS/FAIL per check; exit 1 on any FAIL.
"""
import importlib
import importlib.metadata as md
import sys
import os
import traceback

# macOS stub: radiance_r4d_attn / radiance_vit_attn call sys.setdlopenflags(... | os.RTLD_DEEPBIND),
# a glibc-only flag. Linux (the server) has it; here it is set to 0 so the modules import.
if not hasattr(os, "RTLD_DEEPBIND"):
    os.RTLD_DEEPBIND = 0

results = []


def check(name):
    def deco(fn):
        try:
            out = fn()
            results.append((True, name, out or ""))
        except Exception as e:
            results.append((False, name, f"{e!r}\n{traceback.format_exc(limit=3)}"))
        return fn
    return deco


@check("entry point vllm.general_plugins:radiance_v032 discoverable via dist-info")
def _():
    eps = [e for e in md.entry_points(group="vllm.general_plugins") if e.name == "radiance_v032"]
    assert len(eps) == 1, eps
    assert eps[0].value == "radiance_v032_plugin:register", eps[0].value
    return eps[0].value


@check("load_general_plugins() loads the plugin (VLLM_PLUGINS not set)")
def _():
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    import radiance_v032_plugin as p
    assert "r4d_backend" in p._DONE and "runtime" in p._DONE, p._DONE
    return sorted(p._DONE)


@check("S1b: AttentionBackendEnum.CUSTOM -> radiance_r4d_attn.R4DAttentionBackend")
def _():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    path = AttentionBackendEnum.CUSTOM.get_path()
    assert path == "radiance_r4d_attn.R4DAttentionBackend", path
    cls = AttentionBackendEnum.CUSTOM.get_class()
    assert cls.get_name() == "CUSTOM"
    # 0.32 calls it with the group's spec (vllm/v1/worker/utils.py)
    assert cls.get_supported_kernel_block_sizes(None) == [16]
    assert cls.get_supported_kernel_block_sizes(kv_cache_spec=object()) == [16]
    lay = cls.supported_kv_cache_layouts()
    assert lay is not None and lay[0].name == "LBHNC", lay
    # the CLI string the server will pass
    from vllm.config.attention import AttentionConfig
    assert AttentionConfig.validate_backend_before("CUSTOM") is AttentionBackendEnum.CUSTOM
    return f"{cls.__module__}.{cls.__qualname__}; layouts={lay}"


@check("S1a: MXFP4 proxy at the front of _POSSIBLE_MXFP4_KERNELS[ROCM] after importing kernels.linear")
def _():
    import vllm.model_executor.kernels.linear as L
    from vllm.platforms import PlatformEnum
    ks = L._POSSIBLE_MXFP4_KERNELS[PlatformEnum.ROCM]
    assert ks[0].__name__ == "RadianceMxfp4W4A8LinearKernel", [k.__name__ for k in ks]
    ok, reason = ks[0].is_supported()
    # no HIP extension on this machine: must decline cleanly, not raise
    assert ok is False and reason, (ok, reason)
    return f"{[k.__name__ for k in ks]}; is_supported -> {reason!r}"


@check("S1a: radiance_mxfp4.kernel_class() builds against 0.32 MxFp4LinearKernel (without .so)")
def _():
    import radiance_mxfp4
    radiance_mxfp4._KERNEL_CLS = None
    cls = radiance_mxfp4._make_kernel_class()
    from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearKernel
    assert issubclass(cls, MxFp4LinearKernel)
    for m in ("is_supported", "can_implement", "process_weights_after_loading", "apply_weights"):
        assert m in cls.__dict__, m
    abstract = getattr(MxFp4LinearKernel, "__abstractmethods__", set())
    missing = [m for m in abstract if m not in cls.__dict__]
    assert not missing, missing
    return f"abstract={sorted(abstract)}"


@check("S1c: GDN OOT subclass registered and selected on instantiation")
def _():
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as G
    from vllm.model_executor.custom_op import op_registry_oot
    sub = op_registry_oot.get("QwenGatedDeltaNetAttention")
    assert sub is not None and sub.__name__ == "RadianceQwenGatedDeltaNetAttention", sub
    assert issubclass(sub, G.QwenGatedDeltaNetAttention)
    inst = G.QwenGatedDeltaNetAttention.__new__(G.QwenGatedDeltaNetAttention)
    assert type(inst) is sub, type(inst)
    # the stock methods the subclass relies on still exist with the expected signatures
    import inspect
    sig = list(inspect.signature(G.QwenGatedDeltaNetAttention._forward_core).parameters)
    assert sig == ["self", "mixed_qkv", "b", "a", "core_attn_out"], sig
    for m in ("forward_hip", "forward_cuda", "_forward_core_rocm"):
        assert hasattr(G.QwenGatedDeltaNetAttention, m), m
    return f"{sub.__module__}.{sub.__qualname__}"


@check("S1c: radiance_gdn only reads GDNAttentionMetadata fields that exist in 0.32")
def _():
    import ast
    import dataclasses
    from pathlib import Path
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    fields = {f.name for f in dataclasses.fields(GDNAttentionMetadata)}
    src = Path(importlib.util.find_spec("radiance_gdn").origin).read_text()
    used = {n.attr for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id in ("md", "meta", "attn_metadata")}
    # 'get' = dict access on the raw per-layer metadata; radiance_stash_indices is added only by
    # patch_gdn_lazy (opt-in RADIANCE_GDN_LAZY=1, not part of this port) and read only on that path.
    missing = sorted(used - fields - {"get", "radiance_stash_indices"})
    assert not missing, missing
    return f"{len(used)} fields: {sorted(used)}"


@check("monkeypatch targets of radiance_* exist in 0.32")
def _():
    targets = [
        ("vllm.model_executor.layers.quantization.fp8", "Fp8LinearMethod", "process_weights_after_loading"),
        ("vllm.model_executor.layers.linear", "UnquantizedLinearMethod", "process_weights_after_loading"),
        ("vllm.v1.worker.gpu_worker", "Worker", "compile_or_warm_up_model"),
        ("vllm.distributed.device_communicators.cuda_communicator", "CudaCommunicator", "all_reduce"),
        ("vllm.v1.spec_decode.llm_base_proposer", "SpecDecodeBaseProposer", "_greedy_sample"),
        ("vllm.v1.spec_decode.llm_base_proposer", "SpecDecodeBaseProposer", "propose"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "propose_draft_token_ids"),
        ("vllm.v1.attention.ops.vit_attn_wrappers", None, "apply_sdpa"),
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MTP", "load_weights"),
        ("vllm.model_executor.models.qwen3_dflash2", "DFlash2Qwen3ForCausalLM", "load_weights"),
    ]
    missing, ok = [], []
    for mod, cls, attr in targets:
        try:
            obj = importlib.import_module(mod)
            if cls:
                obj = getattr(obj, cls)
            getattr(obj, attr)
            ok.append(f"{cls or mod}.{attr}")
        except Exception as e:
            missing.append(f"{mod}:{cls}.{attr} ({type(e).__name__}: {e})")
    assert not missing, missing
    return f"{len(ok)} targets OK"


def main():
    bad = 0
    for ok, name, info in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if info:
            for line in str(info).rstrip().splitlines():
                print(f"        {line}")
        bad += not ok
    print(f"\n{len(results) - bad}/{len(results)} PASS")
    sys.exit(1 if bad else 0)


main()
