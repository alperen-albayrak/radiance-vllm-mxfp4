"""Static check: every `from vllm.X import Y` / `import vllm.X` (also inside functions) in the given
python files must resolve against a vLLM source tree: module file exists and Y is defined at
module top level (def/class/assign/import) or is a submodule.

usage: check_symbols.py <vllm_src_root> <file.py> [...]
Exit 1 if anything is unresolved.
"""
import ast
import sys
from pathlib import Path

root = Path(sys.argv[1])
_cache = {}


def mod_path(mod):
    p = root / Path(*mod.split("."))
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    if p.is_dir():
        return p  # namespace pkg
    return None


def top_names(path):
    if path in _cache:
        return _cache[path]
    names = set()
    if path.is_file():
        tree = ast.parse(path.read_text())
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    for e in ast.walk(t):
                        if isinstance(e, ast.Name):
                            names.add(e.id)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and isinstance(n.target, ast.Name):
                names.add(n.target.id)
            elif isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names):
                base = path.parent
                for _ in range(max(n.level - 1, 0)):
                    base = base.parent
                sub = base.joinpath(*(n.module or "").split(".")) if n.level else \
                    root.joinpath(*n.module.split("."))
                cand = sub / "__init__.py" if (sub / "__init__.py").exists() else sub.with_suffix(".py")
                if cand.exists():
                    names |= top_names(cand)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    names.add((a.asname or a.name).split(".")[0])
    _cache[path] = names
    return names


bad = 0
for f in sys.argv[2:]:
    tree = ast.parse(Path(f).read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and \
                (n.module == "vllm" or n.module.startswith("vllm.")):
            mp = mod_path(n.module)
            if mp is None:
                print(f"MISSING-MODULE {f}:{n.lineno} {n.module}")
                bad += 1
                continue
            for a in n.names:
                if a.name == "*":
                    continue
                if a.name in top_names(mp) or mod_path(f"{n.module}.{a.name}"):
                    continue
                print(f"MISSING-NAME   {f}:{n.lineno} from {n.module} import {a.name}")
                bad += 1
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("vllm.") and mod_path(a.name) is None:
                    print(f"MISSING-MODULE {f}:{n.lineno} import {a.name}")
                    bad += 1
print(f"checked {len(sys.argv) - 2} files, unresolved: {bad}")
sys.exit(1 if bad else 0)
