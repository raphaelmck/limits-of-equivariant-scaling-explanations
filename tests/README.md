# tests/

Two checkers, answering different questions.

## `check_manuscript.py` — does the paper agree with the data?

```bash
make check-manuscript TEX=path/to/main.tex
```

Reads every reported number **out of the LaTeX source** and compares it to the regenerated
artifacts: the scaling exponents, both figures' plotted values and captions, all four appendix
tables, the evaluation-population sizes, the compute accounting, and the matched-frontier counts.
Nothing is transcribed by hand, so a number edited in the manuscript but not in the data — or the
reverse — fails here.

Each rule anchors on the surrounding prose or on a table row, so a rule that stops matching is
reported as `NOT FOUND` rather than silently passing. `--show-uncovered` lists the numeric
literals no rule claimed, so a newly added claim cannot slip through unnoticed; the remainder are
structural (`\ell_{\max}=2`, block 9, `n=512`, `95\%`).

The manuscript source is not kept in this repository — pass its path.

## `test_paper_numbers.py` — do the artifacts still hold the submitted values?

```bash
make test
```

Pins the values as of the frozen submission and checks them against the regenerated tables. It
needs no LaTeX source, so it runs anywhere and catches a change in the data. It does **not** track
later edits to the manuscript; `check_manuscript.py` is the authority on paper-versus-data
agreement.

## `make validate` — do the regenerated tables match the frozen ones?

A third, separate question, answered by `scripts/validate.py`: that re-running the analysis
reproduces the frozen tables in `data/`.
