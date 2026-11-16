"""
Provenance — which code produced a result, and whether it still would.

A stored result is only as good as the model that computed it.  When the
model changes, results computed before the change are not wrong, but they
are no longer what the application would answer today, and nothing in the
numbers says so.  So every job result records a fingerprint of the code
behind it, and reading a result compares that with the code now running.

The fingerprint is derived, not declared.  A version number that someone
must remember to bump is forgotten exactly when it matters: on a change
that "obviously doesn't affect results".  Instead, starting from the
module a job type names, this follows the imports within the project's
science packages and hashes the source of every module reached.

Comments, docstrings and formatting do not count (the source is hashed
after a round trip through the parser), so editing them does not mark
results as outdated.  Any other edit does, including ones that happen not
to change the numbers: the error is always on the side of saying
"outdated" too often, never too rarely.

What the fingerprint cannot see: non-Python inputs such as data files, and
the platform's floating-point arithmetic.  Results are bit-for-bit
reproducible on one deployment; across machines they agree to rounding.
"""

import ast
import hashlib
import importlib.util
import json
import os
import platform
from functools import lru_cache
from typing import Optional

import numpy

# Imports are followed only into these packages: the code that decides results
SCIENCE_PACKAGES = ("simulation", "terrain")


def _is_docstring(node: ast.stmt) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and _is_docstring(node.body[0]):
                # A body may not be empty: `pass` stands in for a lone docstring
                node.body = node.body[1:] or [ast.Pass()]


def _module_file(name: str) -> Optional[str]:
    """Path of a module's source file, or None if `name` is not a source module."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return spec.origin


def _imports(tree: ast.AST, module: str, is_package: bool, packages: tuple[str, ...]) -> set[str]:
    """Modules of `packages` that this module imports, anywhere in its body."""
    def in_scope(name: str) -> bool:
        return name.split(".")[0] in packages

    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if in_scope(alias.name))
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                # Relative: climb from this module's package
                parts = module.split(".") if is_package else module.split(".")[:-1]
                parts = parts[:len(parts) - (node.level - 1)]
                base = ".".join(parts + ([base] if base else []))
            if not base or not in_scope(base):
                continue
            found.add(base)
            # `from package import name`: name may itself be a module
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


@lru_cache(maxsize=None)
def module_hashes(root: str, packages: tuple[str, ...] = SCIENCE_PACKAGES) -> dict[str, str]:
    """
    {module name: hash of its normalised source} for `root` and every module
    of `packages` it imports, directly or not.

    Cached: source files do not change under a running process.
    """
    hashes: dict[str, str] = {}
    pending = [root]
    while pending:
        name = pending.pop()
        if name in hashes:
            continue
        path = _module_file(name)
        if path is None:
            if name == root:
                raise ValueError(f"'{root}' is not a module with Python source.")
            # Not a source module: a name imported from a package that turned
            # out not to be a submodule, or a package with no __init__.py
            continue
        with open(path, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        is_package = os.path.basename(path) == "__init__.py"
        pending.extend(_imports(tree, name, is_package, packages))
        # Importing a module runs its package's __init__ first
        if "." in name:
            pending.append(name.rpartition(".")[0])
        _strip_docstrings(tree)
        hashes[name] = hashlib.sha256(ast.unparse(tree).encode()).hexdigest()[:12]
    return dict(sorted(hashes.items()))


def describe(root: str, settings: dict, revision: int,
             packages: tuple[str, ...] = SCIENCE_PACKAGES) -> dict:
    """
    The provenance record stored with a result.

    `fingerprint` is what decides whether a result is outdated.  It covers
    the code, the numpy version (which owns the random number streams),
    the settings and the handler's revision; the remaining fields say what
    went into it, or are for the record only (python, commit).
    """
    modules = module_hashes(root, packages)
    decisive = {"modules": modules, "numpy": numpy.__version__, "settings": settings, "revision": revision}
    digest = hashlib.sha256(json.dumps(decisive, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "fingerprint": digest,
        **decisive,
        "python": platform.python_version(),
        # Set by the host on each deploy; absent in development
        "commit": os.environ.get("RENDER_GIT_COMMIT") or None,
    }
