#!/usr/bin/env python3
"""Apply the Radiance overlay (base: vLLM nightly f8b29fab4, 2026-11-20) to an installed vLLM (site-packages) or a vLLM source tree.

usage: apply_overlay.py [--dry-run] [--allow-skip] [SP]
  SP  directory that contains vllm/ (default: sysconfig purelib of this interpreter)

Stdlib only (the container needs neither git nor patch). All-or-nothing: every file is merged
in memory first; nothing is written unless every file succeeds (see --allow-skip).
Per modified vLLM file:
  * already equal to the patched reference          -> NOOP
  * equal to the f8b29fab4 base                     -> replace by the patched reference
  * otherwise (a build that differs from f8b29fab4)  -> hunk-wise apply of base->patched:
      each hunk's old block (context + removed lines) must occur exactly once in the target;
      if it does not, the context is trimmed step by step (max 2 lines per side, like
      patch --fuzz=2), still requiring a unique match. No match = FAIL for that file.
--allow-skip: a file that fails and is not in REQUIRED is left stock instead of aborting; files in
the same GROUP are then skipped together (the topk-composite hunks pass a new keyword argument
between files, so they must land all or none). Skipped files are listed in
SP/.radiance_v032_backup/SKIPPED.txt, which verify_overlay.py honours.
Originals are saved under SP/.radiance_v032_backup/ before writing.
New files (radiance_*.py, .pth, radiance_v032_plugin/, its dist-info) are copied as is.
Afterwards run verify_overlay.py.
"""
from __future__ import annotations

import ast
import difflib
import shutil
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_FUZZ = 2
# Needed for the S1 route (MXFP4 W4A8, R4D attention + GDN); never skipped.
REQUIRED = {
    "vllm/_aiter_ops.py",
    "vllm/model_executor/kernels/linear/mxfp4/aiter.py",
    "vllm/platforms/rocm.py",
    "vllm/third_party/flash_linear_attention/ops/chunk.py",
}
# All-or-none groups (cross-file call signatures).
GROUPS = {
    "topk_composite": {
        "vllm/v1/sample/metadata.py",
        "vllm/v1/sample/ops/topk_topp_sampler.py",
        "vllm/v1/sample/rejection_sampler.py",
        "vllm/v1/worker/gpu/sample/sampler.py",
        "vllm/v1/worker/gpu/sample/states.py",
        "vllm/v1/worker/gpu_input_batch.py",
    },
}


def load_manifest():
    rows = []
    for line in (HERE / "manifest.tsv").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        kind, path, markers, sha = line.split("\t")
        rows.append((kind, path, int(markers), sha))
    return rows


def hunks(base: list[str], patched: list[str]):
    """(old_block, new_block, n_leading_ctx, n_trailing_ctx) per hunk, 3 lines of context."""
    sm = difflib.SequenceMatcher(a=base, b=patched, autojunk=False)
    for group in sm.get_grouped_opcodes(3):
        old, new = [], []
        for tag, i1, i2, j1, j2 in group:
            old.extend(base[i1:i2])
            new.extend(patched[j1:j2])
        lead = group[0][2] - group[0][1] if group[0][0] == "equal" else 0
        trail = group[-1][2] - group[-1][1] if group[-1][0] == "equal" else 0
        yield old, new, lead, trail


def find_unique(hay: list[str], needle: list[str]) -> int | None:
    if not needle:
        return None
    n, hits = len(needle), []
    first = needle[0]
    for i in range(len(hay) - n + 1):
        if hay[i] == first and hay[i:i + n] == needle:
            hits.append(i)
            if len(hits) > 1:
                return None
    return hits[0] if hits else None


def already_applied(target: str, base: str, patched: str) -> bool:
    """True when every hunk's new block (trimmed like merge()) is already in target."""
    t = target.splitlines(keepends=True)
    b = base.splitlines(keepends=True)
    p = patched.splitlines(keepends=True)
    for old, new, lead, trail in hunks(b, p):
        if len(new) - lead - trail <= 0:
            # pure deletion: "applied" means the removed block is gone
            if find_unique(t, old) is not None:
                return False
            continue
        for fuzz in range(MAX_FUZZ + 1):
            cl, ct = min(fuzz, lead), min(fuzz, trail)
            nw = new[cl:len(new) - ct]
            if nw and find_unique(t, nw) is not None:
                break
        else:
            return False
    return True


def merge(target: str, base: str, patched: str) -> tuple[str | None, str]:
    t = target.splitlines(keepends=True)
    b = base.splitlines(keepends=True)
    p = patched.splitlines(keepends=True)
    applied = []
    for k, (old, new, lead, trail) in enumerate(hunks(b, p)):
        done = False
        for fuzz in range(MAX_FUZZ + 1):
            cl, ct = min(fuzz, lead), min(fuzz, trail)
            o = old[cl:len(old) - ct]
            nw = new[cl:len(new) - ct]
            i = find_unique(t, o)
            if i is not None:
                applied.append((i, len(o), nw, k, fuzz))
                t = t[:i] + nw + t[i + len(o):]
                done = True
                break
        if not done:
            return None, f"hunk {k + 1}: old block not found exactly once (fuzz<= {MAX_FUZZ})"
    out = "".join(t)
    try:
        ast.parse(out)
    except SyntaxError as e:
        return None, f"merged file does not parse: {e}"
    fz = [a[4] for a in applied if a[4]]
    return out, f"{len(applied)} hunks" + (f", fuzz used {fz}" if fz else "")


def main(argv):
    dry = "--dry-run" in argv
    allow_skip = "--allow-skip" in argv
    args = [a for a in argv if not a.startswith("--")]
    sp = Path(args[0]) if args else Path(sysconfig.get_paths()["purelib"])
    if not (sp / "vllm").is_dir():
        raise SystemExit(f"FAIL: {sp}/vllm not found")
    rows = load_manifest()
    plan, failures = [], []
    for kind, path, _markers, _sha in rows:
        if kind != "mod":
            continue
        dst = sp / path
        if not dst.exists():
            failures.append((path, "target file missing"))
            continue
        cur = dst.read_text()
        base = (HERE / "base" / path).read_text()
        pat = (HERE / "patched" / path).read_text()
        if cur == pat:
            print(f"  NOOP   {path}")
            continue
        if cur == base:
            plan.append((path, pat))
            print(f"  BASE   {path} (identical to f8b29fab4 -> patched copy)")
            continue
        if already_applied(cur, base, pat):
            print(f"  NOOP   {path} (hunks already present)")
            continue
        merged, info = merge(cur, base, pat)
        if merged is None:
            failures.append((path, info))
            print(f"  FAIL   {path}: {info}")
        else:
            plan.append((path, merged))
            print(f"  MERGE  {path}: {info}")
    skipped: set[str] = set()
    if failures and allow_skip and not any(p in REQUIRED for p, _ in failures):
        skipped = {p for p, _ in failures}
        for members in GROUPS.values():
            if skipped & members:
                skipped |= members
        plan = [(p, t) for p, t in plan if p not in skipped]
        for p in sorted(skipped):
            print(f"  SKIP   {p} (stays stock)")
        failures = []
    if failures:
        print(f"\n{len(failures)} file(s) not applicable; nothing written.")
        for p, why in failures:
            print(f"   - {p}: {why}{'  [REQUIRED]' if p in REQUIRED else ''}")
        return 1
    if dry:
        print("\n--dry-run: nothing written.")
        return 0
    bak = sp / ".radiance_v032_backup"
    for path, text in plan:
        dst = sp / path
        b = bak / path
        if not b.exists():
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, b)
        dst.write_text(text)
    n_new = 0
    for kind, path, _m, _s in rows:
        if kind == "new":
            src, dst = HERE / "files" / path, sp / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_new += 1
    bak.mkdir(parents=True, exist_ok=True)
    (bak / "SKIPPED.txt").write_text("".join(f"{p}\n" for p in sorted(skipped)))
    print(f"\nwritten: {len(plan)} modified vLLM files, {n_new} new files, "
          f"{len(skipped)} skipped; backup in {bak}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
