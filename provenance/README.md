# provenance/

Where the code under `pipeline/` came from, and how to check it.

- **`pipeline_manifest.json`** — one entry per released file: its original path, and the SHA256 of
  both the original and the released copy. Also records the model patch and what it does.
- **`check_pipeline.py`** — the verifier.
- **`rename_table.json`** — the substitutions the release applied. This is the only file that
  contains pre-anonymization paths and names, so it is **excluded from the anonymous release**
  and restored on de-anonymization.

## Anonymization note

This manifest deliberately **withholds the source revision** (repository name, branch, and commit
hash) the code under `pipeline/` was taken from. A private development repository's name, branch,
or commit identifier can itself be identifying or searchable, independent of anything the files
contain — for instance, a personal branch name. Withholding it costs nothing functionally: every
file's integrity is fully checkable here by SHA256, and its logical identity to the original is
fully checkable by `check_pipeline.py` given local access to the original source tree. The
withheld revision is recorded privately, outside this repository, and will be restored here after
acceptance.

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
`<PIPELINE_ROOT>`), module and identifier names carrying internal study labels, module
docstrings, and the source-revision identifier described above. Four message strings were also
rewritten; they are enumerated in `rename_table.json` under `text_substitutions` rather than left
implicit. No logic was changed, and the checker is what makes that claim falsifiable.

## Unversioned originals

Six files under `pipeline/evaluation_pools/` were copied from a working directory that was not
under version control at the time, so their manifest entries carry no revision either way. Their
SHA256 is recorded, and the copies here are now the versioned record of that code.
