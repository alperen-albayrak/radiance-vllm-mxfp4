#!/usr/bin/env python3
"""Verify an applied Radiance overlay (base: vLLM nightly f8b29fab4, 2026-11-20).

usage: verify_overlay.py [SP]
Checks (no vLLM import, no GPU needed):
  1. every manifest file exists in SP and compiles (compile(), .py only)
  2. per file the number of lines mentioning 'radiance' equals the manifest ("anchors")
  3. new files are byte-identical to the overlay copy
  4. the vllm.general_plugins entry point radiance_v032 is discoverable from SP
  5. radiance_v032_plugin imports (it does not import vLLM at module level)
For the vLLM-level checks (registrations really land) run selftest_overlay.py with SP on
PYTHONPATH (needs an importable vLLM; on the server: inside the container).
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv):
    sp = Path(argv[0]) if argv else Path(sysconfig.get_paths()["purelib"])
    bad = 0
    n_mod = n_new = n_skip = 0
    sk = sp / ".radiance_v032_backup" / "SKIPPED.txt"
    skipped = set(sk.read_text().split()) if sk.exists() else set()
    for line in (HERE / "manifest.tsv").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        kind, path, markers, sha = line.split("\t")
        f = sp / path
        if path in skipped:
            print(f"SKIP  {path} (left stock by apply_overlay.py --allow-skip)")
            n_skip += 1
            continue
        if not f.exists():
            print(f"FAIL  missing: {path}")
            bad += 1
            continue
        if path.endswith(".py"):
            try:  # compile() rather than py_compile: writes no __pycache__ into SP
                compile(f.read_text(), str(f), "exec")
            except SyntaxError as e:
                print(f"FAIL  does not compile: {path}: {e}")
                bad += 1
        text = f.read_text(errors="replace")
        got = sum("radiance" in ln.lower() for ln in text.splitlines())
        if got != int(markers):
            print(f"FAIL  anchors {path}: {got} != expected {markers}")
            bad += 1
        if kind == "new":
            n_new += 1
            if hashlib.sha256(f.read_bytes()).hexdigest() != sha:
                print(f"FAIL  new file differs: {path}")
                bad += 1
        else:
            n_mod += 1
    # entry point, resolved in a clean interpreter with SP first on sys.path
    code = ("import importlib.metadata as m,sys;"
            "e=[x for x in m.entry_points(group='vllm.general_plugins') if x.name=='radiance_v032'];"
            "print(e[0].value if len(e)==1 else 'MISSING:'+repr(e));"
            "import radiance_v032_plugin as p;print('plugin-import-ok', p.R4D_BACKEND_PATH)")
    r = subprocess.run([sys.executable, "-I", "-c", f"import sys;sys.path.insert(0,{str(sp)!r});{code}"],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0 or "MISSING" in out or "plugin-import-ok" not in out:
        print(f"FAIL  entry point / plugin import:\n{out}")
        bad += 1
    else:
        print(f"OK    entry point + plugin import: {out.splitlines()[0]}")
    print(f"{'OK' if not bad else 'FAIL'}  {n_mod} modified + {n_new} new files checked, "
          f"{n_skip} skipped, {bad} error(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
