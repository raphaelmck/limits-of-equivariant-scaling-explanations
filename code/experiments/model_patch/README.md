# Model instrumentation patch

`per_block_hidden_states.patch` is the complete set of modifications made to the upstream model
implementations for this work. It applies to `nets/uma/backbone.py`, `nets/uma/model.py`,
`nets/egnn/model.py`, and `nets/faenet/model.py` in the training repository.

**The patch is purely additive instrumentation.** Every change appends a tensor to an output
dictionary or accumulates per-block hidden states into a list. No forward computation, weight,
prediction, energy, or force output is altered by it: running the patched model and the unpatched
model on the same input produces identical energies and forces. Its only effect is to make
intermediate representations reachable from outside the model, which is what the layerwise
analyses need.

The interventions themselves do not rely on this patch. `pipeline/intervention/esen_intervention.py`
reproduces the backbone's forward orchestration and calls each block directly, so it reaches the
per-block state without the patched return values.

Applied to the source revision recorded in `provenance/pipeline_manifest.json`
(`upstream_commit`), this patch reconstructs the exact model code the reported results were
produced with.
