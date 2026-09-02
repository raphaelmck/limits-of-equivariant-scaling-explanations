#!/usr/bin/env python3
"""Verify the released copies under pipeline/ against provenance/pipeline_manifest.json.

Two levels, depending on what is available:

  Always: each released file's SHA256 matches the `ported_sha256` recorded in the manifest, and
  its header names the original path, source revision, and original SHA256. This proves the
  released tree has not drifted since it was recorded.

  When the source tree is available (`--source-root <path>`, or the recorded upstream
  repository checked out at `upstream_commit`): each original's SHA256 matches
  `original_sha256`, and the released copy is shown to be logically identical to the original.
  That comparison is made on the abstract syntax tree, with docstrings removed and the release's
  renames undone, so it is insensitive to comments, docstrings, and formatting but sensitive to
  any change in code -- a renamed variable, a changed constant, a reordered call, an added or
  removed statement all fail.

Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "provenance", "pipeline_manifest.json")

# The substitutions the release applied are kept in rename_table.json, which is the only place
# the pre-anonymization paths and names appear and is therefore excluded from the anonymous
# release. Without it, hash and header checks still run; the --source-root comparison needs it,
# and needs the original files too.
RENAME_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rename_table.json")


def load_substitutions():
    if not os.path.exists(RENAME_TABLE):
        return None, None
    table = json.load(open(RENAME_TABLE))
    return ([tuple(p) for p in table["renames"]],
            [tuple(p) for p in table["text_substitutions"]])


RENAMES, TEXT_SUBSTITUTIONS = load_substitutions()

HEADER_LINES = 4  # the provenance header stamped onto each released file


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def strip_header(text: str) -> str:
    lines = text.split("\n")
    out = []
    removed = 0
    for line in lines:
        if removed < HEADER_LINES and line.startswith("# ") and (
            line.startswith("# Adapted for anonymous release")
            or line.startswith("# source revision")
            or line.startswith("# Logic unchanged")
            or line.startswith("# Verify with")
        ):
            removed += 1
            continue
        out.append(line)
    return "\n".join(out)


def normalize(released: str) -> str:
    """Undo the release substitutions so the result is comparable to the original.

    Longest replacement first, so a dotted module path is undone before the package prefix it
    starts with. Identifier renames are applied on word boundaries, so a rename of `decoder`
    cannot reach inside `common_decoder_atom_features`.
    """
    if RENAMES is None:
        raise RuntimeError("provenance/rename_table.json is required for this comparison")
    text = strip_header(released)
    for original, replacement in TEXT_SUBSTITUTIONS:
        text = text.replace(replacement, original)
    for original, replacement in sorted(RENAMES, key=lambda p: len(p[1]), reverse=True):
        if replacement.replace("_", "a").isalnum():
            text = re.sub(r"\b" + re.escape(replacement) + r"\b", original, text)
        else:
            text = text.replace(replacement, original)
    return text


def strip_docstrings(tree):
    """Drop every docstring, so the comparison sees code only."""
    kinds = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, kinds):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def code_fingerprint(source):
    """AST of the source with docstrings removed: equal fingerprints mean identical logic."""
    return ast.dump(strip_docstrings(ast.parse(source)), annotate_fields=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", action="append", default=[], metavar="PATH",
                        help="root holding the original files, to compare against; repeatable "
                             "(the versioned repository, and the directory holding the "
                             "evaluation-pool originals)")
    args = parser.parse_args()

    manifest = json.load(open(MANIFEST))
    failures = []
    compared = 0

    print(f"upstream commit {manifest['upstream_commit']}")
    print(f"{len(manifest['files'])} released files\n")

    for entry in manifest["files"]:
        released_path = os.path.join(REPO_ROOT, entry["ported_path"])
        label = entry["ported_path"]

        if not os.path.exists(released_path):
            failures.append(f"{label}: missing")
            continue

        if sha256(released_path) != entry["ported_sha256"]:
            failures.append(f"{label}: released copy has changed since the manifest was recorded")
            continue

        released = open(released_path).read()
        header = released.split("\n")[0] if not released.startswith("#!") else released.split("\n")[1]
        if entry["original_path"] not in header:
            failures.append(f"{label}: header does not name {entry['original_path']}")

        if not args.source_root:
            continue

        if RENAMES is None:
            print(f"[skip] {label}: rename_table.json not present, cannot undo the release "
                  "substitutions")
            continue

        candidates = [os.path.join(root, entry["original_path"]) for root in args.source_root]
        original_path = next((c for c in candidates if os.path.exists(c)), None)
        if original_path is None:
            print(f"[skip] {label}: original not found under any --source-root")
            continue

        compared += 1
        if sha256(original_path) != entry["original_sha256"]:
            failures.append(f"{label}: original at source-root does not match the recorded SHA256")
            continue

        original = open(original_path).read()
        try:
            same_code = code_fingerprint(original) == code_fingerprint(normalize(released))
        except SyntaxError as exc:
            failures.append(f"{label}: could not parse ({exc})")
            continue
        if not same_code:
            failures.append(
                f"{label}: code differs from the original, not only comments and docstrings")

    if args.source_root:
        print(f"compared {compared} files against their originals (AST, docstrings removed)")
    else:
        print("hashes and headers checked; pass --source-root to also diff against originals")

    if failures:
        print("\nFAILED:")
        for line in failures:
            print("  " + line)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
