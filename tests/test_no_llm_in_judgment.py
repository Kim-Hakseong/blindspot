"""Rule A2 -- no LLM anywhere on the judgment path.

Metrics, boundary location, and cost accounting decide what the report claims.
If a model can influence them, the report is an opinion rather than a
measurement. This test walks the *transitive* import graph of those packages
and fails if any LLM client is reachable, so the guarantee cannot be broken by
an indirect import three modules deep.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "blindspot"

# Packages that decide pass/fail, where the boundary is, and what it cost.
JUDGMENT_PACKAGES = ["metrics", "boundary", "cost", "degrade", "measure"]

# Import roots that would put a model on the judgment path.
FORBIDDEN_ROOTS = {
    "anthropic",
    "openai",
    "boto3",  # reaches Bedrock; judgment code must not call AWS at all
    "botocore",
    "langchain",
    "llama_index",
    "transformers",
    "mcp",
    "google",
    "cohere",
    "ollama",
    "litellm",
}

# In-repo modules that are allowed to hold an LLM client.
LLM_OWNING_PACKAGES = {"blindspot.agent"}


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports_of(path: pathlib.Path) -> set[str]:
    """Every module this file imports, absolute and relative resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    this_mod = _module_name(path)
    pkg_parts = this_mod.split(".")
    if path.name != "__init__.py":
        pkg_parts = pkg_parts[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)] if node.level > 1 else pkg_parts
                target = list(base) + ([node.module] if node.module else [])
                prefix = ".".join(target)
            else:
                prefix = node.module or ""
            for alias in node.names:
                found.add(f"{prefix}.{alias.name}" if prefix else alias.name)
                if prefix:
                    found.add(prefix)
    return found


def _file_for(module: str) -> pathlib.Path | None:
    base = SRC / pathlib.Path(*module.split("."))
    if (p := base.with_suffix(".py")).is_file():
        return p
    if (p := base / "__init__.py").is_file():
        return p
    return None


def _reachable_from(package: str) -> dict[str, list[str]]:
    """Transitive in-repo closure -> {module: import chain that reached it}."""
    root = PKG / package
    seen: dict[str, list[str]] = {}
    stack = [(_module_name(p), [_module_name(p)]) for p in sorted(root.rglob("*.py"))]
    while stack:
        mod, chain = stack.pop()
        if mod in seen:
            continue
        seen[mod] = chain
        path = _file_for(mod)
        if path is None:
            continue
        for imported in _imports_of(path):
            if imported.startswith("blindspot") and imported not in seen:
                if _file_for(imported) is not None:
                    stack.append((imported, chain + [imported]))
    return seen


@pytest.mark.parametrize("package", JUDGMENT_PACKAGES)
def test_judgment_package_exists(package):
    assert (PKG / package).is_dir(), f"missing judgment package: {package}"


@pytest.mark.parametrize("package", JUDGMENT_PACKAGES)
def test_no_llm_client_reachable_from_judgment_code(package):
    for module, chain in sorted(_reachable_from(package).items()):
        path = _file_for(module)
        if path is None:
            continue
        for imported in sorted(_imports_of(path)):
            root = imported.split(".")[0]
            assert root not in FORBIDDEN_ROOTS, (
                f"{module} imports {imported!r}, putting an LLM/cloud client on the "
                f"judgment path. Import chain: {' -> '.join(chain)}"
            )


@pytest.mark.parametrize("package", JUDGMENT_PACKAGES)
def test_judgment_code_does_not_reach_the_agent_package(package):
    for module, chain in sorted(_reachable_from(package).items()):
        for owner in LLM_OWNING_PACKAGES:
            assert not module.startswith(owner), (
                f"judgment package {package!r} reaches {module!r}. "
                f"Import chain: {' -> '.join(chain)}"
            )


def test_the_guard_would_actually_catch_a_violation(tmp_path, monkeypatch):
    """A test that cannot fail is not a guarantee. Prove the detector detects."""
    offender = tmp_path / "blindspot" / "metrics"
    offender.mkdir(parents=True)
    (tmp_path / "blindspot" / "__init__.py").write_text("")
    (offender / "__init__.py").write_text("")
    (offender / "bad.py").write_text("import anthropic\n")

    monkeypatch.setattr(f"{__name__}.SRC", tmp_path)
    monkeypatch.setattr(f"{__name__}.PKG", tmp_path / "blindspot")

    with pytest.raises(AssertionError, match="judgment path"):
        test_no_llm_client_reachable_from_judgment_code("metrics")
