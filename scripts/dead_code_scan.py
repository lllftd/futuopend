#!/usr/bin/env python3
"""
AST scan for unreachable code (conservative heuristics).

1) Top-level defs: name count across *.py <= 1 (with blocklist).
2) Nested defs: FunctionDef whose *immediate* enclosing scope is another
   FunctionDef (ClassDef on stack so SDK hooks on nested classes are ignored).
3) Private class methods (leading _): no Attribute(self|cls|super()., _m) in file,
   and name is not a method on any resolved base class (overrides are excluded).

Does not catch: dynamic getattr, decorators that register methods, tests.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIR = {".venv", "venv", "__pycache__", ".git", "node_modules"}

TOP_BLOCKLIST = frozenset({"main", "parse_args", "pytest_configure"})
MAGIC_METHODS = {
    n
    for n in dir(object)
    if n.startswith("__") and n.endswith("__")
} | {"__class_getitem__", "__init_subclass__"}

RISKY_DECORATORS = frozenset(
    {"fixture", "parametrize", "register", "abstractmethod", "cached_property"}
)


def iter_py_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        if any(x in p.parts for x in SKIP_DIR):
            continue
        yield p


def load_modules(root: Path) -> dict[Path, ast.Module]:
    out: dict[Path, ast.Module] = {}
    for p in iter_py_files(root):
        try:
            out[p] = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
    return out


def class_method_sets(tree: ast.Module) -> dict[str, set[str]]:
    """class_name -> method names defined on that class in this module."""
    m: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(item.name)
            m[node.name] = names
    return m


def parse_imports(tree: ast.Module) -> tuple[dict[str, str], dict[str, str]]:
    """
    simple_from: LocalName -> 'pkg.mod'
    simple_mod: alias -> 'pkg.mod' for 'import pkg.mod as alias'
    """
    simple_from: dict[str, str] = {}
    simple_mod: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module and node.names:
                mod = node.module
                for al in node.names:
                    if al.name == "*":
                        continue
                    simple_from[al.asname or al.name] = mod
        elif isinstance(node, ast.Import):
            for al in node.names:
                base = al.name.split(".")[0]
                simple_mod[al.asname or base] = al.name
    return simple_from, simple_mod


def resolve_module_file(from_file: Path, dotted: str) -> Path | None:
    """Map import module string to file under root."""
    parts = dotted.split(".")
    # package: live.monitor -> live/monitor.py
    cand = ROOT.joinpath(*parts)
    if cand.is_dir():
        cand = cand / "__init__.py"
    else:
        cand = cand.with_suffix(".py")
    if cand.is_file():
        return cand
    # try relative to importing file (same package)
    parent_pkg = from_file.parent
    rel = parent_pkg.relative_to(ROOT)
    rel_parts = list(rel.parts)
    for i in range(len(rel_parts), -1, -1):
        prefix = rel_parts[:i]
        trial = ROOT.joinpath(*prefix, *parts)
        if trial.is_dir():
            trial = trial / "__init__.py"
        else:
            trial = trial.with_suffix(".py")
        if trial.is_file():
            return trial
    return None


def resolve_base_class_methods(
    from_file: Path,
    tree: ast.Module,
    class_node: ast.ClassDef,
    global_classes: dict[tuple[Path, str], set[str]],
) -> set[str]:
    """Union of method names defined on resolved base classes (best-effort)."""
    methods: set[str] = set()
    simple_from, simple_mod = parse_imports(tree)
    for base in class_node.bases:
        mod_file: Path | None = None
        cls_name = ""
        if isinstance(base, ast.Name):
            cls_name = base.id
            if cls_name in simple_from:
                mf = resolve_module_file(from_file, simple_from[cls_name])
                if mf and (mf, cls_name) in global_classes:
                    methods |= global_classes[mf, cls_name]
            elif cls_name in simple_mod:
                # import live.monitor as M then M.SymbolMonitor — skip
                pass
        elif isinstance(base, ast.Attribute):
            # live.monitor.SymbolMonitor
            parts: list[str] = []
            n: ast.expr = base
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value  # type: ignore
            if isinstance(n, ast.Name):
                parts.append(n.id)
            parts.reverse()
            if len(parts) >= 2:
                cls_name = parts[-1]
                mod_dots = ".".join(parts[:-1])
                mf = resolve_module_file(from_file, mod_dots)
                if mf and (mf, cls_name) in global_classes:
                    methods |= global_classes[mf, cls_name]
    return methods


def decorator_name(d: ast.expr) -> str:
    if isinstance(d, ast.Name):
        return d.id
    if isinstance(d, ast.Attribute):
        return d.attr
    if isinstance(d, ast.Call):
        return decorator_name(d.func)
    return ""


def method_risky_decorators(m: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for d in m.decorator_list:
        n = decorator_name(d)
        if n in RISKY_DECORATORS or n.startswith("pytest"):
            return True
    return False


# --- nested: only FunctionDef directly inside another FunctionDef ---

def nested_dead_funcs(tree: ast.Module) -> list[tuple[int, str]]:
    dead: list[tuple[int, str]] = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[ast.AST] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._func(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._func(node)

        def _func(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
            parent = self.stack[-1] if self.stack else None
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _name_loaded_in(parent, node.name, skip=node):
                    dead.append((node.lineno, node.name))
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

    V().visit(tree)
    return dead


def _name_loaded_in(
    parent: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
    skip: ast.AST,
) -> bool:
    class R(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Name(self, n: ast.Name):
            if isinstance(n.ctx, ast.Load) and n.id == name:
                self.found = True

        def visit_FunctionDef(self, n: ast.FunctionDef):
            if n is skip:
                return
            self.generic_visit(n)

        def visit_AsyncFunctionDef(self, n: ast.AsyncFunctionDef):
            if n is skip:
                return
            self.generic_visit(n)

        def visit_ClassDef(self, n: ast.ClassDef):
            # definitions inside nested classes: ignore for enclosing-function refs
            self.generic_visit(n)

    r = R()
    r.visit(parent)
    return r.found


# --- private methods ---

def _attr_on_self_super_cls(node: ast.Attribute, method: str) -> bool:
    if node.attr != method:
        return False
    v = node.value
    if isinstance(v, ast.Name) and v.id in ("self", "cls"):
        return True
    if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super":
        return True
    return False


def method_used_in_file(tree: ast.Module, class_name: str, method: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == method:
            if _attr_on_self_super_cls(node, method):
                return True
            if isinstance(node.value, ast.Name) and node.value.id == class_name:
                return True
    return False


def scan_private_method_dead(
    path: Path,
    tree: ast.Module,
    global_classes: dict[tuple[Path, str], set[str]],
) -> list[tuple[int, str, str, str]]:
    out: list[tuple[int, str, str, str]] = []
    base_methods = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_methods[node.name] = resolve_base_class_methods(
            path, tree, node, global_classes
        )

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        cname = node.name
        inherited = base_methods.get(cname, set())
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            m = item.name
            if m in MAGIC_METHODS:
                continue
            if not m.startswith("_") or m.startswith("__"):
                continue  # public or dunder
            if method_risky_decorators(item):
                continue
            if m in inherited:
                continue  # likely override / same name as base
            if not method_used_in_file(tree, cname, m):
                out.append(
                    (item.lineno, cname, m, "private method: no self./cls./super(). ref in file")
                )
    return out


def top_level_dead_by_count(root: Path) -> list[tuple[str, int, str]]:
    texts: list[tuple[Path, str]] = []
    for p in iter_py_files(root):
        try:
            texts.append((p, p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            pass

    def count_word(name: str) -> int:
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        return sum(len(pat.findall(t)) for _, t in texts)

    dead = []
    for p, src in texts:
        try:
            mod = ast.parse(src)
        except SyntaxError:
            continue
        for n in mod.body:
            if not isinstance(n, ast.FunctionDef):
                continue
            if n.name.startswith("test_") or n.name in TOP_BLOCKLIST:
                continue
            if count_word(n.name) <= 1:
                dead.append((str(p.relative_to(root)), n.lineno, n.name))
    return dead


def main() -> int:
    global ROOT
    ap = argparse.ArgumentParser(description="Scan for likely dead functions (heuristic).")
    ap.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repo root (default: parent of scripts/)",
    )
    args = ap.parse_args()
    ROOT = args.root.resolve()

    modules = load_modules(ROOT)
    global_classes: dict[tuple[Path, str], set[str]] = {}
    for p, tree in modules.items():
        for cname, mset in class_method_sets(tree).items():
            global_classes[p, cname] = mset

    print("=== Top-level def: repo-wide name count <= 1 ===")
    tl = top_level_dead_by_count(ROOT)
    for f, ln, name in tl:
        print(f"  {f}:{ln}  {name}")
    if not tl:
        print("  (none)")

    print("\n=== Nested def: function inside function, never referenced in parent ===")
    all_nested: list[tuple[str, int, str]] = []
    for p, tree in modules.items():
        for ln, name in nested_dead_funcs(tree):
            all_nested.append((str(p.relative_to(ROOT)), ln, name))
    for f, ln, name in sorted(all_nested):
        print(f"  {f}:{ln}  {name}")
    if not all_nested:
        print("  (none)")

    print("\n=== Private class methods: no self./cls./super(). in file, not base override ===")
    all_pm: list[tuple[str, int, str, str, str]] = []
    for p, tree in modules.items():
        for ln, c, m, reason in scan_private_method_dead(p, tree, global_classes):
            all_pm.append((str(p.relative_to(ROOT)), ln, c, m, reason))
    for row in sorted(all_pm):
        print(f"  {row[0]}:{row[1]}  {row[2]}.{row[3]}  ({row[4]})")
    if not all_pm:
        print("  (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
