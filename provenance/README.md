# provenance/

Where the code under `pipeline/` came from, and how to check it.

- **`pipeline_manifest.json`** — one entry per released file: its original path, the source
  revision it was taken at, and the SHA256 of both the original and the released copy. Also
  records the model patch and what it does.
- **`check_pipeline.py`** — the verifier.
- **`rename_table.json`** — the substitutions the release applied. This is the only file that
  contains pre-anonymization paths and names, so it is **excluded from the anonymous release**
  and restored on de-anonymization.

## Checking

```bash
python3 provenance/check_pipeline.py
```

Verifies that every released file matches the SHA256 recorded for it and carries a header naming
its original. This proves the released tree has not drifted since the manifest was written, and
runs anywhere.

```bash
python3 provenance/check_pipeline.py --source-root <analysis-repo> --source-root <eval-dir>
```

Additionally verifies each original against its recorded SHA256 and proves the released copy is
*logically identical* to it. The comparison is made on abstract syntax trees with docstrings
removed and the release's renames undone: comments, docstrings, and formatting may differ, code
may not — a renamed variable, changed constant, reordered call, or added statement all fail.
Needs both the original files and `rename_table.json`.

## What the release changed

Import paths, filesystem paths (replaced by `<CHECKPOINT_ROOT>`, `<DATA_ROOT>`, `<PROJECT_ROOT>`,
`<PIPELINE_ROOT>`), module and identifier names carrying internal study labels, and module
docstrings. Four message strings were also rewritten; they are enumerated in `rename_table.json`
under `text_substitutions` rather than left implicit. No logic was changed, and the checker is
what makes that claim falsifiable.

## Unversioned originals

Six files under `pipeline/evaluation_pools/` were copied from a working directory that was not
under version control at the time, so their manifest entries carry no `upstream_commit`. Their
SHA256 is recorded, and the copies here are now the versioned record of that code.
