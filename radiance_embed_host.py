"""RADIANCE embed-host: the target's input embedding in pinned host memory, read over PCIe.

RADIANCE_EMBED_HOST=1 (default 0). Called by patch_embed_host.py at the end of
BaseModelLoader.load_model; serve-mxfp4.sh copies this module into site-packages.

WHY THIS TABLE. `embed_tokens` is [vocab, hidden] bf16 -- 248320 x 5120 = 2.37 GiB on Qwen3.8-27B,
13% of the one-card footprint -- and a forward pass reads ONE ROW of it per token, 10 KB. That
makes it the one large tensor that can leave VRAM without the GPU streaming it back: a decode step
touches ~16 rows (target verify plus the drafter's block), a 2560-token prefill chunk 26 MB.
`lm_head` is the same size and the opposite case -- every step multiplies all of it -- which is
why a tied embedding is refused outright.

HOW. The Parameter's storage is swapped for a UVA (zero-copy) device view of a pinned host copy
(vLLM's get_accelerator_view_from_cpu_tensor; cuda_view.cu is in the HIP build too). The copy is
handed over UNPINNED on purpose: for a pageable tensor the helper makes an exact-size
cudaHostAlloc(Mapped) buffer owned by the view's deleter, whereas torch's pin_memory() goes
through the caching host allocator, which rounds up to a power of two and would pin 4 GiB for
this 2.37 GiB table. Host RAM peaks at twice the table during the copy, then holds it once.
The gather kernel then reads rows across PCIe from device code: no host sync, no input_ids round
trip, nothing a captured graph cannot replay. The Parameter object keeps its identity, so the
DFlash drafter -- which shares the target's module (`draft_inner.embed_tokens = target_embed` in
the dflash loader) -- follows it with no change of its own. The VRAM copy is released before the
loader returns, and vLLM's ROCm DeviceMemoryProfiler reads the CURRENT allocation
(reset_peak_memory_stats, then max_memory_allocated), so the smaller model is what it records: a
profiled KV cache gets the difference automatically, a pinned one via calibrate-kv.sh.

WHAT IT NEVER DOES is fall back to vLLM's non-UVA offload, which re-uploads the whole parameter on
every forward (2.37 GiB per step, ~200 ms on a Gen4 x8 link). If the view cannot be made, or does
not read back bit-identical, the GPU copy stays and serving continues exactly as without the flag.

At startup it logs the gather cost through the view against the VRAM copy, at the decode width
and at the prefill chunk, so the PCIe price on the host at hand is read off the log, not assumed.

MEASURED 2026-09-22, one R9700 at TP=1 (MAXSEQS=4, CHUNK=2560, dflash SPEC=7, Gen4 x8 link):
model 20.0 -> 17.63 GiB, KV pool 136,441 -> 199,920 tokens, served ceiling 130,240 -> 196,608.
Bulk readback through the view 12.1 GB/s; gather 16 rows 9.4 -> 21.6 us and 2560 rows 44.6 ->
1811.8 us, i.e. +12 us on a ~45 ms decode step and +1.8 ms on a ~1.1 s prefill chunk. Cold prefill
at 124K measured 2,283 t/s against 2,280 before the change, and 2,008 t/s cold at 170K. The drafter
does follow the module: the 2.37 GiB comes off the load exactly once.
"""
import os
import sys
import time

import torch

ENABLED = os.environ.get("RADIANCE_EMBED_HOST", "0") == "1"
# Rows gathered per timed call: the decode width (MAXSEQS x (SPEC+1) is at most 64 on the shipped
# shapes; 16 is one request's target verify plus its drafter block) and the prefill chunk.
DECODE_ROWS = int(os.environ.get("RADIANCE_EMBED_HOST_DECODE_ROWS", "16"))
BENCH = os.environ.get("RADIANCE_EMBED_HOST_BENCH", "1") == "1"
# Rows per slice of the bit-exact readback. The comparison allocates a bool mask of the slice, so
# the whole-table compare (2.37 GiB, 1.2 G elements) is never materialised at once.
_CHECK_ROWS = 16384

_offloaded = set()


def _say(msg: str) -> None:
    print(f"[radiance] embed-host: {msg}", file=sys.stderr, flush=True)


def _gib(nbytes: float) -> str:
    return f"{nbytes / 1073741824:.2f} GiB"


def _view_fn():
    """vLLM's UVA view constructor, under either of the names it has shipped with."""
    try:
        from vllm.utils import torch_utils as tu
    except Exception:
        return None
    for name in ("get_accelerator_view_from_cpu_tensor", "get_cuda_view_from_cpu_tensor"):
        fn = getattr(tu, name, None)
        if fn is not None:
            return fn
    return None


def _target_embedding(model):
    lm = model.get_language_model() if hasattr(model, "get_language_model") else model
    inner = getattr(lm, "model", None)
    return lm, getattr(inner, "embed_tokens", None) if inner is not None else None


def _refusal(model, lm, emb):
    """None if `emb` can move to host memory, else the reason it cannot."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )

    if emb is None:
        return "no language_model.model.embed_tokens on this model"
    if not isinstance(emb, VocabParallelEmbedding) or isinstance(emb, ParallelLMHead):
        return f"embed_tokens is {type(emb).__name__}, not a VocabParallelEmbedding"
    if not isinstance(getattr(emb, "quant_method", None), UnquantizedEmbeddingMethod):
        return f"embed_tokens is quantized ({type(emb.quant_method).__name__})"
    w = getattr(emb, "weight", None)
    if w is None or not w.is_cuda or w.dim() != 2 or not w.is_contiguous():
        return "embed_tokens.weight is not a contiguous 2-D device tensor"
    if w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"embed_tokens.weight dtype {w.dtype}"
    # A tied lm_head would put the per-step GEMM on the PCIe link. Checked by storage, not by
    # config, so a tie made at load time is caught too.
    ptr = w.data_ptr()
    sharers = [n for n, p in model.named_parameters(remove_duplicate=False) if p.data_ptr() == ptr]
    if len(sharers) > 1:
        return f"embed_tokens storage is shared with {sharers} (tied lm_head?)"
    return None


def _time_gather(weight, idx, iters=20):
    """Mean of `iters` gathers in microseconds, after 3 warm-ups."""
    for _ in range(3):
        torch.nn.functional.embedding(idx, weight)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.nn.functional.embedding(idx, weight)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def maybe_offload(model, vllm_config, model_config) -> None:
    """Move the TARGET model's embed_tokens to pinned host memory. Drafter loads are skipped:
    the dflash loader replaces the drafter's table with the target's module afterwards."""
    if not ENABLED:
        return
    if model_config is not vllm_config.model_config:
        return
    if id(model) in _offloaded:
        return
    lm, emb = _target_embedding(model)
    why = _refusal(model, lm, emb)
    if why is not None:
        _say(f"NOT applied -- {why}; embed_tokens stays in VRAM")
        return
    to_view = _view_fn()
    if to_view is None:
        _say("NOT applied -- this vLLM has no UVA view constructor; embed_tokens stays in VRAM")
        return

    w = emb.weight
    dev = w.device
    nbytes = w.numel() * w.element_size()
    try:
        t0 = time.perf_counter()
        torch.cuda.synchronize(dev)
        free_before = torch.cuda.mem_get_info(dev)[0]
        # Pageable on purpose -- see HOW in the module docstring: the view pins an exact-size copy.
        view = to_view(w.to("cpu"))
        if view.device != dev or view.shape != w.shape or view.dtype != w.dtype:
            _say(f"NOT applied -- view is {view.device}/{tuple(view.shape)}/{view.dtype}, "
                 f"weight {dev}/{tuple(w.shape)}/{w.dtype}")
            return
        # Bit-exact readback of the whole table THROUGH the view, i.e. through the same device-side
        # PCIe path the gather kernel will use -- which also measures that path's bulk rate.
        torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        rows = w.shape[0]
        for r in range(0, rows, _CHECK_ROWS):
            if not torch.equal(view[r:r + _CHECK_ROWS], w[r:r + _CHECK_ROWS]):
                _say(f"NOT applied -- readback through the view differs at rows {r}.."
                     f"{min(r + _CHECK_ROWS, rows) - 1}; embed_tokens stays in VRAM")
                return
        torch.cuda.synchronize(dev)
        t_check = time.perf_counter() - t1

        bench = ""
        if BENCH:
            chunk = int(getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 0) or 2048)
            parts = []
            for n in (DECODE_ROWS, chunk):
                idx = torch.randint(0, rows, (n,), device=dev)
                g, h = _time_gather(w, idx), _time_gather(view, idx)
                parts.append(f"{n} rows {g:.1f} -> {h:.1f} us")
            bench = "; gather VRAM -> host: " + ", ".join(parts)
    except Exception as e:  # noqa: BLE001 -- any failure before the swap leaves VRAM serving
        _say(f"NOT applied -- {type(e).__name__}: {e}; embed_tokens stays in VRAM")
        return

    w.data = view  # same Parameter object, so the drafter's share follows; the view owns the pin
    del view
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    freed = torch.cuda.mem_get_info(dev)[0] - free_before

    _offloaded.add(id(model))
    _say(f"{tuple(w.shape)} {str(w.dtype).replace('torch.', '')} ({_gib(nbytes)}) moved to pinned "
         f"host memory, VRAM freed {_gib(freed)}; bit-exact readback over PCIe "
         f"{nbytes / t_check / 1e9:.1f} GB/s{bench}; {time.perf_counter() - t0:.1f} s")
