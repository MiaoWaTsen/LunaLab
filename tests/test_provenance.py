"""
Tests for backend/provenance.py: the fingerprint must change when the code
that decides a result changes, and only then.

Each test writes a small package to a temporary directory and fingerprints
it, so nothing here depends on the real science modules staying as they are.
"""

import importlib
import sys
import textwrap

import pytest

from backend import provenance

MODEL = """
    from fp_pkg.physics import energy

    def crater(mass, speed):
        return energy(mass, speed) ** 0.25
"""
PHYSICS = """
    def energy(mass, speed):
        return 0.5 * mass * speed ** 2
"""


@pytest.fixture
def package(tmp_path, monkeypatch):
    """Write modules into a throwaway package and fingerprint it."""
    root = tmp_path / "fp_pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    class Package:
        def write(self, files=None, **modules):
            """write(model="..."), or write({"sub/__init__": "..."}) for nested paths."""
            for name, source in {**(files or {}), **modules}.items():
                path = root / f"{name}.py"
                path.parent.mkdir(exist_ok=True)
                path.write_text(textwrap.dedent(source))
            # Forget what was read before: the files have changed
            provenance.module_hashes.cache_clear()
            importlib.invalidate_caches()
            for loaded in [m for m in sys.modules if m.split(".")[0] == "fp_pkg"]:
                del sys.modules[loaded]

        def hashes(self, root_module="fp_pkg.model"):
            return provenance.module_hashes(root_module, ("fp_pkg",))

        def fingerprint(self, settings=None, revision=1):
            return provenance.describe("fp_pkg.model", settings or {}, revision, ("fp_pkg",))["fingerprint"]

    pkg = Package()
    pkg.write(model=MODEL, physics=PHYSICS)
    yield pkg
    provenance.module_hashes.cache_clear()
    for loaded in [m for m in sys.modules if m.split(".")[0] == "fp_pkg"]:
        del sys.modules[loaded]


# ---------------------------------------------------------------------------
# What is covered
# ---------------------------------------------------------------------------

ALWAYS = ["fp_pkg", "fp_pkg.model"]   # the root and the package whose __init__ runs first


def test_imported_modules_are_followed(package):
    assert list(package.hashes()) == [*ALWAYS, "fp_pkg.physics"]


def test_a_package_init_counts(package):
    before = package.fingerprint()
    package.write({"__init__": "import math\nmath.pi = 3\n"})   # runs before any module in it
    assert package.fingerprint() != before


def test_imports_are_followed_transitively(package):
    package.write(physics="from fp_pkg.constants import HALF\n" + textwrap.dedent(PHYSICS),
                  constants="HALF = 0.5\n")
    assert "fp_pkg.constants" in package.hashes()


@pytest.mark.parametrize("statement", [
    "import fp_pkg.extra",
    "from fp_pkg import extra",
    "from fp_pkg.extra import VALUE",
    "from . import extra",
    "from .extra import VALUE",
])
def test_every_form_of_import_is_followed(package, statement):
    package.write(model=f"{statement}\n", extra="VALUE = 1\n")
    assert "fp_pkg.extra" in package.hashes()


def test_an_import_inside_a_function_is_followed(package):
    package.write(model="def late():\n    from fp_pkg.extra import VALUE\n    return VALUE\n",
                  extra="VALUE = 1\n")
    assert "fp_pkg.extra" in package.hashes()


def test_relative_import_from_a_subpackage_climbs_correctly(package):
    package.write({"sub/__init__": "", "sub/deep": "from ..physics import energy\n"},
                  model="from fp_pkg.sub import deep\n")
    hashes = package.hashes()
    assert {"fp_pkg.sub", "fp_pkg.sub.deep", "fp_pkg.physics"} <= set(hashes)


def test_modules_outside_the_named_packages_are_not_followed(package):
    package.write(model="import json\nimport numpy\n" + textwrap.dedent(MODEL))
    assert list(package.hashes()) == [*ALWAYS, "fp_pkg.physics"]


def test_circular_imports_terminate(package):
    package.write(model="import fp_pkg.physics\n", physics="import fp_pkg.model\n")
    assert list(package.hashes()) == [*ALWAYS, "fp_pkg.physics"]


def test_root_that_is_not_a_module_is_an_error(package):
    with pytest.raises(ValueError, match="fp_pkg.missing"):
        package.hashes("fp_pkg.missing")


# ---------------------------------------------------------------------------
# What changes the fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_is_stable(package):
    first = package.fingerprint()
    package.write(model=MODEL, physics=PHYSICS)   # rewritten, unchanged
    assert package.fingerprint() == first


def test_changing_the_named_module_changes_it(package):
    before = package.fingerprint()
    package.write(model=MODEL.replace("0.25", "0.26"))
    assert package.fingerprint() != before


def test_changing_a_module_it_imports_changes_it(package):
    before = package.fingerprint()
    package.write(physics=PHYSICS.replace("0.5", "0.4"))
    assert package.fingerprint() != before


def test_changing_an_unrelated_module_does_not(package):
    before = package.fingerprint()
    package.write(unrelated="ANSWER = 42\n")
    assert package.fingerprint() == before


def test_comments_docstrings_and_formatting_do_not_change_it(package):
    before = package.fingerprint()
    package.write(physics='''
        """Kinetic energy, now documented."""


        def energy( mass,speed ):   # reformatted
            """Half m v squared."""
            # an explanation
            return (0.5*mass*speed**2)
    ''')
    assert package.fingerprint() == before


def test_a_string_that_is_not_a_docstring_counts(package):
    before = package.fingerprint()
    package.write(physics=textwrap.dedent(PHYSICS) + "UNIT = 'J'\n")
    assert package.fingerprint() != before


def test_a_function_that_was_only_a_docstring_still_parses(package):
    package.write(physics='def energy(mass, speed):\n    """Not written yet."""\n')
    assert package.fingerprint()


def test_settings_and_revision_change_it(package):
    base = package.fingerprint({"samples": 1000}, revision=1)
    assert package.fingerprint({"samples": 2000}, revision=1) != base
    assert package.fingerprint({"samples": 1000}, revision=2) != base
    assert package.fingerprint({"samples": 1000}, revision=1) == base


def test_numpy_version_changes_it(package, monkeypatch):
    before = package.fingerprint()
    monkeypatch.setattr(provenance.numpy, "__version__", "99.0.0")
    assert package.fingerprint() != before


def test_python_version_and_commit_are_recorded_but_do_not_change_it(package, monkeypatch):
    before = provenance.describe("fp_pkg.model", {}, 1, ("fp_pkg",))
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc123")
    monkeypatch.setattr(provenance.platform, "python_version", lambda: "9.9.9")
    after = provenance.describe("fp_pkg.model", {}, 1, ("fp_pkg",))
    assert (after["commit"], after["python"]) == ("abc123", "9.9.9")
    assert before["commit"] is None
    assert after["fingerprint"] == before["fingerprint"]


# ---------------------------------------------------------------------------
# The real job type
# ---------------------------------------------------------------------------

def test_inference_batch_covers_the_model_it_runs():
    provenance.module_hashes.cache_clear()
    modules = provenance.module_hashes("simulation.inference")
    assert {"simulation.inference", "simulation.crater_model"} <= set(modules)
    assert all(name.split(".")[0] in provenance.SCIENCE_PACKAGES for name in modules)
