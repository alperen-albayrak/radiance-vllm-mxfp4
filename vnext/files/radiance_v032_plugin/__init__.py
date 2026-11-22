"""Radiance (GGZ14/vllm-mxfp4) integration for vLLM 0.32 as a `vllm.general_plugins` plugin.

Replaces four text patches of the 0.29 patchset with registrations through vLLM's own
extension points. vLLM calls `register()` once per process (API server, engine core, every
worker) via load_general_plugins(); everything here is idempotent and env-gated by the same
RADIANCE_* switches the radiance modules already read.

  S1a  MXFP4 W4A8 linear kernel   -> _POSSIBLE_MXFP4_KERNELS[PlatformEnum.ROCM] (was patch_quark_mxfp4 hunk 1)
  S1b  R4D attention backend      -> register_backend(AttentionBackendEnum.CUSTOM, ...) (was patch_r4d enum hunk)
                                     select with  --attention-backend CUSTOM
  S1c  R4D gated-delta-net hook   -> PluggableLayer OOT subclass of QwenGatedDeltaNetAttention
                                     (was patch_r4d _forward_core hunk + patch_gdn_glue)
  --   radiance runtime hooks     -> radiance_kernels.install_all() (was install_radiance_hooks.py)

Set RADIANCE_V032_PLUGIN=0 to make register() a no-op.
"""
from __future__ import annotations

import os
import sys

from ._hooks import when_imported

R4D_BACKEND_PATH = "radiance_r4d_attn.R4DAttentionBackend"
MXFP4_PROXY_NAME = "RadianceMxfp4W4A8LinearKernel"
_DONE: set[str] = set()


def _say(msg: str) -> None:
    sys.stderr.write(f"[radiance.v032] {msg}\n")


# ---------------------------------------------------------------- S1b: R4D attention backend
def register_r4d_backend() -> None:
    """--attention-backend CUSTOM -> radiance_r4d_attn.R4DAttentionBackend.

    CUSTOM rather than a new "R4D" enum member: a member created at run time does not survive
    pickling into a spawned engine/worker process (Enum unpickles by value before any plugin has
    run), CUSTOM is a static member and only its class path lives in the per-process override
    table, which this plugin fills in every process.
    """
    if "r4d_backend" in _DONE:
        return
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.CUSTOM, R4D_BACKEND_PATH)
    _DONE.add("r4d_backend")


# ------------------------------------------------------------- S1a: MXFP4 W4A8 linear kernel
def _real_mxfp4_kernel():
    import radiance_mxfp4  # loads radiance_mxfp4_fp8.so; only reached in the worker

    return radiance_mxfp4.kernel_class()


def _install_mxfp4(linear_mod) -> None:
    if "mxfp4" in _DONE:
        return
    from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearKernel
    from vllm.platforms import PlatformEnum

    class RadianceMxfp4W4A8LinearKernel(MxFp4LinearKernel):
        """Lazy proxy for radiance_mxfp4.kernel_class().

        The real class is built on first use (is_supported() at model load, in the worker), so
        registering it never imports the HIP extension in the API-server process.
        Instantiation returns an instance of the real class.
        """

        def __new__(cls, *args, **kwargs):
            real = _real_mxfp4_kernel()
            if real is None:
                raise RuntimeError("radiance MXFP4 W4A8 kernel class unavailable")
            return real(*args, **kwargs)

        @classmethod
        def is_supported(cls, compute_capability=None):
            try:
                real = _real_mxfp4_kernel()
            except Exception as e:
                return False, f"radiance_mxfp4 unavailable: {e!r}"
            if real is None:
                return False, "radiance_mxfp4.kernel_class() returned None"
            return real.is_supported(compute_capability)

        @classmethod
        def can_implement(cls, config):
            real = _real_mxfp4_kernel()
            if real is None:
                return False, "radiance_mxfp4.kernel_class() returned None"
            return real.can_implement(config)

    table = linear_mod._POSSIBLE_MXFP4_KERNELS
    kernels = table.setdefault(PlatformEnum.ROCM, [])
    if not any(getattr(k, "__name__", "") == MXFP4_PROXY_NAME for k in kernels):
        kernels.insert(0, RadianceMxfp4W4A8LinearKernel)
    _DONE.add("mxfp4")


def register_mxfp4_kernel() -> None:
    when_imported("vllm.model_executor.kernels.linear", _install_mxfp4)


# ------------------------------------------------------------ S1c: R4D gated-delta-net hook
def _radiance_gdn_active():
    """radiance_gdn when the all-R4D GDN path is on (RADIANCE_USE_R4D=1 and r4d loaded)."""
    try:
        import radiance_gdn
    except Exception:
        return None
    return radiance_gdn if getattr(radiance_gdn, "ALL", False) else None


_GDN_DECLINE = os.environ.get("RADIANCE_GDN_DECLINE", "")


def _gdn_declined(md) -> bool:
    if not _GDN_DECLINE:
        return False
    if _GDN_DECLINE == "all":
        return True
    prefill = getattr(md, "num_prefills", 0) > 0
    return prefill if _GDN_DECLINE == "prefill" else not prefill


def _install_gdn(gdn_mod) -> None:
    if "gdn" in _DONE:
        return
    Base = gdn_mod.QwenGatedDeltaNetAttention

    class RadianceQwenGatedDeltaNetAttention(Base):
        """QwenGatedDeltaNetAttention with the libr4d fused GDN prologue on _forward_core.

        Two 0.32 paths would bypass _forward_core and with it the hook:
          * forward_hip -> _forward_core_rocm when aiter's RDNA GDN Triton kernels are
            available (VLLM_ROCM_USE_AITER=1 on gfx12);
          * forward_cuda -> qwen_gdn_attention_core_fused_norm_packed when the fused CUDA
            spec-decode kernel is enabled (VLLM_GDN_DECODE_KERNEL=cuda and the op is built).
        With R4D active both are steered to the plain forward_cuda -> _forward_core path.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._radiance_gdn = _radiance_gdn_active()
            if self._radiance_gdn is None:
                return
            if getattr(self, "enable_fused_gdn_spec_decode", False):
                self.enable_fused_gdn_spec_decode = False
            if getattr(self, "_forward_method", None) is not None and \
                    getattr(self._forward_method, "__func__", None) is Base.forward_hip:
                self._forward_method = self.forward_cuda
            # vLLM after 0.32: its FLA/FlashInfer prefill kernels write the prefix-cache state
            # checkpoints themselves, mid-prefill. The R4D prefill does not, and with the
            # alignment left set no checkpoint is ever written: every prefix-cache hit then
            # restores stale state (wrong tokens, measured 2026-11-22, tests/cachecorrect*.py).
            # None is vLLM's contract for "cannot export": the scheduler splits the prefill
            # at block boundaries and copies the state, as 0.27/0.32 did.
            if getattr(self, "_checkpoint_alignment", None) is not None:
                self._checkpoint_alignment = None
                if "gdn_ckpt_logged" not in _DONE:
                    _DONE.add("gdn_ckpt_logged")
                    _say("R4D GDN: prefill checkpoints not exported -> scheduler splits the "
                         "prefill at prefix-cache block boundaries")
            if "gdn_logged" not in _DONE:
                _DONE.add("gdn_logged")
                _say("R4D GDN active: forward_hip/aiter and fused-norm spec decode steered "
                     "to forward_cuda -> _forward_core (+ R4D prologue)")

        def _forward_core(self, mixed_qkv, b, a, core_attn_out):
            rg = self._radiance_gdn
            if rg is not None:
                md = gdn_mod.get_forward_context().attn_metadata
                if isinstance(md, dict):
                    md = md.get(self.prefix)
                # md None = warm-up/profiling: the stock body handles it.
                # RADIANCE_GDN_DECLINE=prefill|decode|all is a diagnostic: leave that kind of
                # step to vLLM's own core, to tell an R4D path problem from a caching one.
                if md is not None and not _gdn_declined(md) and rg.forward_core_fused(
                        self, mixed_qkv, b, a, core_attn_out):
                    return
            return super()._forward_core(mixed_qkv, b, a, core_attn_out)

    # vLLM's out-of-tree mechanism: instantiating the in-tree class yields this subclass.
    Base.register_oot(RadianceQwenGatedDeltaNetAttention)
    _DONE.add("gdn")


def register_gdn_hook() -> None:
    when_imported("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn", _install_gdn)


# ------------------------------------------------- DFlash2 FP8 drafter: conv layers stay BF16
def _install_dflash2_conv_bf16(mod) -> None:
    """FALLBACK (RADIANCE_DFLASH2_CONV_BF16=1); by default _install_dflash_ignored_layers_offset does this job.
    The FP8 drafter has kernel_projection in BF16, but upstream passes the FP8 config through and
    the exclusion (layers.0..4) does not match the real prefix (layers.64..68), so the layer
    loads as FP8 without a scale (acceptance 0). GGZ14's own code uses quant_config=None."""
    if "dflash2_conv" in _DONE or os.environ.get("RADIANCE_DFLASH2_CONV_BF16", "0") != "1":
        return
    cls = mod.DFlashGroupedConv
    orig = cls.__init__

    def patched(self, *args, **kwargs):
        kwargs["quant_config"] = None
        if len(args) > 6:
            args = args[:6]
        return orig(self, *args, **kwargs)

    cls.__init__ = patched
    _DONE.add("dflash2_conv")
    _say("DFlash2 grouped conv: quant_config=None (BF16)")


def _install_dflash_ignored_layers_offset(mod) -> None:
    """Same offset as exclude_modules, but for Fp8Config.ignored_layers, including the `model.` prefix.
    Equivalent to vllm-project/vllm#61010 (issue #61003); can be removed once that is in vLLM. Disable: RADIANCE_DFLASH2_EXCL=0."""
    if "dflash_excl" in _DONE or os.environ.get("RADIANCE_DFLASH2_EXCL", "1") == "0":
        return
    import re

    pattern = getattr(mod, "_DRAFT_LAYER_PATTERN", None) or re.compile(
        r"(?<![A-Za-z0-9_.])layers\.(\d+)"
    )

    def add_offset(quant_config, start_layer_id, num_hidden_layers):
        ign = getattr(quant_config, "ignored_layers", None)
        if quant_config is None or start_layer_id == 0 or not isinstance(ign, (list, set)):
            return
        add = []
        for name in tuple(ign):
            new = pattern.sub(
                lambda m: m.group(0) if int(m.group(1)) >= num_hidden_layers
                else f"layers.{int(m.group(1)) + start_layer_id}", name)
            if new == name:
                continue
            for cand in (new, f"model.{new}") if "*" not in name else (new,):
                if cand not in ign and cand not in add:
                    add.append(cand)
        for n in add:
            ign.append(n) if isinstance(ign, list) else ign.add(n)
        _say(f"dflash ignored_layers offset +{start_layer_id}: {len(add)} added")

    if hasattr(mod, "_add_global_draft_layer_exclusions"):  # vLLM 0.32
        orig = mod._add_global_draft_layer_exclusions

        def patched(quant_config, start_layer_id, num_hidden_layers):
            orig(quant_config, start_layer_id, num_hidden_layers)
            add_offset(quant_config, start_layer_id, num_hidden_layers)

        mod._add_global_draft_layer_exclusions = patched
    else:  # later vLLM: models.utils.map_draft_quant_config, imported into this module
        orig = mod.map_draft_quant_config

        def patched(quant_config, weight_mapper, prefix, start_layer_id, num_hidden_layers):
            orig(quant_config, weight_mapper, prefix, start_layer_id, num_hidden_layers)
            add_offset(quant_config, start_layer_id, num_hidden_layers)

        mod.map_draft_quant_config = patched
    _DONE.add("dflash_excl")


def register_dflash2_conv_hook() -> None:
    when_imported("vllm.model_executor.models.qwen3_dflash2", _install_dflash2_conv_bf16)
    when_imported("vllm.model_executor.models.qwen3_dflash", _install_dflash_ignored_layers_offset)


# ------------------------------------------------------------------ radiance runtime hooks
def install_runtime_hooks() -> None:
    if "runtime" in _DONE:
        return
    _DONE.add("runtime")
    import radiance_kernels

    radiance_kernels.install_all()


def register() -> None:
    """vllm.general_plugins entry point."""
    if os.environ.get("RADIANCE_V032_PLUGIN", "1") == "0":
        return
    for step in (register_r4d_backend, register_mxfp4_kernel, register_gdn_hook,
                 register_dflash2_conv_hook, install_runtime_hooks):
        try:
            step()
        except Exception as e:
            _say(f"{step.__name__} failed: {e!r}")
