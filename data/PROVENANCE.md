# Provenance conventions

Every frozen table under `data/` carries a `source_file` column (or an equivalent field) naming
the upstream output it was copied from, at full precision and without recomputation. Those paths
use two neutral prefixes:

| Prefix | What it refers to |
|---|---|
| `upstream-training/` | the training and force-evaluation repository: model families, training runs, the per-checkpoint force-scaling table, and the FLOP accounting |
| `upstream-analysis/` | the kernel and intervention analysis repository: force tangent kernel construction, KRR, the irreducible-representation interventions, and the evaluation-pool builders |

Under each, `analysis_outputs/<study>/` is one study's output directory and `analysis_scripts/`
holds the script that produced it. Study directory names are retained so that a given number can
be traced to the specific run that produced it.

Filesystem paths are written against placeholders rather than absolute locations:

| Placeholder | Meaning |
|---|---|
| `<CHECKPOINT_ROOT>` | the root of the checkpoint store, as used in `data/checkpoints.csv` and the frontier tables |
| `<DATA_ROOT>` | the root of the OMol25 dataset splits |
| `<PROJECT_ROOT>` | the directory containing the repositories above |
| `<PIPELINE_ROOT>` | the analysis repository itself, when a path is written relative to its own root |
| `<REPO_ROOT>` | this repository |

Neither upstream repository is required to run anything here: `make analysis` derives every
reported number from the files in this directory alone.
