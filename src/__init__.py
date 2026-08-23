"""Reusable analysis functions for paper_repro Stage 2.

Pure Python standard library only (no numpy/pandas/scipy) — this was a deliberate choice
documented in AUDIT_STAGE2.md, driven by the execution environment for this pass not having
those packages installed/installable. Every module here is import-only: no file I/O and no
side effects happen at import time. All file reads live in scripts/claim1/ and scripts/claim2/.
"""
