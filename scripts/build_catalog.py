#!/usr/bin/env python3
"""Generate notebooks/catalog.json from @app.function definitions in nb*.py files.

Run from the repo root:
    python scripts/build_catalog.py

Or via uv (no install needed):
    uv run scripts/build_catalog.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import ast
import inspect
import json
import textwrap
from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
OUTPUT = NOTEBOOKS_DIR / "catalog.json"

# Canonical import source for functions that exist in multiple notebooks.
# When two notebooks define the same function, prefer the one listed here.
CANONICAL = {
    "load_profiles": "nb01_retrieve_profiles",
    "sample_with_negcon": "nb02_add_metadata",
    "load_similarity_matrix": "nb07_compound_neighborhood",
}

# Semantically equivalent functions with different names across notebooks.
# Listed as {preferred: [aliases...]} — always import the preferred form.
ALIASES: dict[str, list[str]] = {
    "load_similarity_matrix": ["load_sim_cached"],
    "pairwise_cosine_labeled": ["pairwise_cosine_by_plate"],
}


def _annotation_str(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _arg_str(arg: ast.arg) -> str:
    base = arg.arg
    if arg.annotation:
        base += f": {_annotation_str(arg.annotation)}"
    return base


def _default_str(default: ast.expr) -> str:
    return ast.unparse(default)


def extract_functions(path: Path) -> list[dict]:
    source = path.read_text()
    tree = ast.parse(source)
    module = path.stem  # e.g. "nb01_retrieve_profiles"

    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        # Only capture functions decorated with @app.function
        is_app_function = any(
            (isinstance(d, ast.Attribute) and d.attr == "function")
            or (isinstance(d, ast.Name) and d.id == "function")
            for d in node.decorator_list
        )
        if not is_app_function:
            continue

        # Signature: positional args with optional defaults
        args = node.args
        n_args = len(args.args)
        n_defaults = len(args.defaults)
        # defaults align to the END of args list
        padding = n_args - n_defaults

        sig_parts = []
        for i, arg in enumerate(args.args):
            part = _arg_str(arg)
            default_idx = i - padding
            if default_idx >= 0:
                part += f" = {_default_str(args.defaults[default_idx])}"
            sig_parts.append(part)

        # *args and **kwargs
        if args.vararg:
            sig_parts.append(f"*{_arg_str(args.vararg)}")
        if args.kwarg:
            sig_parts.append(f"**{_arg_str(args.kwarg)}")

        ret = _annotation_str(node.returns)
        sig = f"({', '.join(sig_parts)})"
        if ret:
            sig += f" -> {ret}"

        # Docstring
        docstring = ast.get_docstring(node)

        functions.append({
            "name": node.name,
            "module": module,
            "signature": sig,
            "docstring": docstring or "",
        })

    return functions


def build_catalog() -> dict:
    all_functions: list[dict] = []
    seen: dict[str, str] = {}  # name -> first module that defined it

    for nb_path in sorted(NOTEBOOKS_DIR.glob("nb*.py")):
        for fn in extract_functions(nb_path):
            name = fn["name"]
            canonical_module = CANONICAL.get(name)

            if name in seen:
                # Already recorded — skip unless this module is the canonical one
                if canonical_module and fn["module"] == canonical_module:
                    # Replace the earlier non-canonical entry
                    all_functions = [f for f in all_functions if f["name"] != name]
                    fn["canonical"] = True
                    fn["also_defined_in"] = [seen[name]]
                    all_functions.append(fn)
                    seen[name] = fn["module"]
                else:
                    # Mark existing entry as having a duplicate
                    for existing in all_functions:
                        if existing["name"] == name:
                            existing.setdefault("also_defined_in", []).append(fn["module"])
            else:
                if canonical_module and fn["module"] == canonical_module:
                    fn["canonical"] = True
                seen[name] = fn["module"]
                all_functions.append(fn)

    # Group by module for the per-notebook index
    by_module: dict[str, list[dict]] = {}
    for fn in all_functions:
        by_module.setdefault(fn["module"], []).append(fn)

    # Annotate alias functions with a "prefer" pointer
    alias_lookup: dict[str, str] = {}  # alias_name -> preferred_name
    for preferred, aliases in ALIASES.items():
        for alias in aliases:
            alias_lookup[alias] = preferred

    for fn in all_functions:
        if fn["name"] in alias_lookup:
            fn["prefer_instead"] = alias_lookup[fn["name"]]
        if fn["name"] in ALIASES:
            fn["aliases"] = ALIASES[fn["name"]]

    return {
        "version": 1,
        "description": (
            "Machine-readable index of every @app.function in the jx notebook catalog. "
            "Generated by scripts/build_catalog.py — do not edit by hand."
        ),
        "canonical_sources": CANONICAL,
        "aliases": ALIASES,
        "functions": all_functions,
        "by_module": by_module,
    }


def main() -> None:
    catalog = build_catalog()
    OUTPUT.write_text(json.dumps(catalog, indent=2) + "\n")
    total = len(catalog["functions"])
    modules = len(catalog["by_module"])
    duplicates = sum(1 for f in catalog["functions"] if "also_defined_in" in f)
    print(f"Wrote {OUTPUT.relative_to(Path.cwd())} — {total} functions across {modules} notebooks")
    if duplicates:
        print(f"  {duplicates} function(s) defined in multiple notebooks (see 'also_defined_in'):")
        for fn in catalog["functions"]:
            if "also_defined_in" in fn:
                others = ", ".join(fn["also_defined_in"])
                canonical = " [canonical]" if fn.get("canonical") else ""
                print(f"    {fn['name']} in {fn['module']}{canonical}, also in {others}")

    aliases_count = sum(1 for f in catalog["functions"] if "prefer_instead" in f)
    if aliases_count:
        print(f"  {aliases_count} function(s) are semantic aliases (see 'prefer_instead'):")
        for fn in catalog["functions"]:
            if "prefer_instead" in fn:
                print(f"    {fn['name']} ({fn['module']}) → prefer {fn['prefer_instead']}")


if __name__ == "__main__":
    main()
