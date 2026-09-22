#!/usr/bin/env python3
"""Hook radiance_embed_host into the end of BaseModelLoader.load_model.

With RADIANCE_EMBED_HOST=1 the TARGET model's embed_tokens (2.37 GiB bf16 on Qwen3.8-27B) moves to
pinned host memory behind a UVA device view once its weights are loaded and processed, and before
the DFlash loader shares that module into the drafter. The whole decision -- target vs drafter,
tied or quantized tables, a view that does not read back bit-exact -- lives in the module; this
patch only calls it. With the flag unset the inserted branch is one environment lookup per model
load, and the loaded model is byte-identical.

The anchor is the tail of the loader's body: every checkpoint format the shipped launchers serve
goes through BaseModelLoader.load_model (DefaultModelLoader does not override it). The import is
inside the branch and NOT guarded: a serve that asked for the flag and cannot import the module
must fail at load, not silently keep the table in VRAM.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
LOADER = SP / "vllm/model_executor/model_loader/base_loader.py"

ANCHOR = (
    "            process_weights_after_loading(model, model_config, target_device)\n"
    "\n"
    "        return model.eval()\n"
)
NEW = (
    "            process_weights_after_loading(model, model_config, target_device)\n"
    "\n"
    "            # --- RADIANCE embed-host (patch_embed_host.py) ---\n"
    "            if __import__(\"os\").environ.get(\"RADIANCE_EMBED_HOST\", \"0\") == \"1\":\n"
    "                import radiance_embed_host as _radiance_embed_host\n"
    "\n"
    "                _radiance_embed_host.maybe_offload(model, vllm_config, model_config)\n"
    "\n"
    "        return model.eval()\n"
)


def main():
    apply(LOADER, ANCHOR, NEW, "RADIANCE embed-host (patch_embed_host.py)",
          "embed_tokens -> pinned host memory behind a UVA view (RADIANCE_EMBED_HOST=1)")


if __name__ == "__main__":
    main()
