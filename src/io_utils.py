"""Thin CSV/JSON/JSONL loading helpers shared by scripts/claim1 and scripts/claim2.

No path in this module or anywhere else in src/ or scripts/ may point outside
paper_repro/data/ or paper_repro/manifests/ (see AUDIT_STAGE2.md hard-scope constraint).
"""
from __future__ import annotations

import csv
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MANIFESTS_DIR = os.path.join(REPO_ROOT, "manifests")
ANALYSIS_OUT_DIR = os.path.join(REPO_ROOT, "analysis_out")


def data_path(*parts: str) -> str:
    return os.path.join(DATA_DIR, *parts)


def manifest_path(*parts: str) -> str:
    return os.path.join(MANIFESTS_DIR, *parts)


def out_path(*parts: str) -> str:
    return os.path.join(ANALYSIS_OUT_DIR, *parts)


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(x, default=None):
    if x is None or x == "":
        return default
    return float(x)
