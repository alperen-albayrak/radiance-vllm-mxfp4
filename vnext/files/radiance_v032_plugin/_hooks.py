"""Post-import hooks: run a callback once a module has finished importing.

Used so the plugin can register into vLLM modules (the linear-kernel table, the GDN layer)
without importing them itself. Importing those eagerly in the API-server process would pull in
aiter and initialise HIP, which forces the engine core to spawn instead of fork (vLLM keeps
those imports lazy for the same reason).
"""
from __future__ import annotations

import importlib.abc
import sys
from collections.abc import Callable

_PENDING: dict[str, list[Callable]] = {}


def _run(cb: Callable, module) -> None:
    try:
        cb(module)
    except Exception as e:  # never break the import of a vLLM module
        sys.stderr.write(f"[radiance.v032] hook {getattr(cb, '__name__', cb)!r} "
                         f"on {module.__name__} failed: {e!r}\n")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _PENDING:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        orig_exec = loader.exec_module

        def exec_module(module, _orig=orig_exec, _name=name):
            _orig(module)
            for cb in _PENDING.pop(_name, []):
                _run(cb, module)

        loader.exec_module = exec_module
        return spec


_FINDER = _Finder()


def when_imported(name: str, cb: Callable) -> None:
    """Call cb(module) now if `name` is fully imported, else right after its import."""
    mod = sys.modules.get(name)
    spec = getattr(mod, "__spec__", None)
    if mod is not None and not getattr(spec, "_initializing", False):
        _run(cb, mod)
        return
    _PENDING.setdefault(name, []).append(cb)
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
