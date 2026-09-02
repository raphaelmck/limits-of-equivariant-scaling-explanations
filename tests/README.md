# tests/

`test_paper_numbers.py` checks every number the paper reports -- exponents, both figures' plotted
values, all four appendix tables, the evaluation-population sizes, and the matched-frontier counts
-- against the tables `make analysis` regenerates. Each check names the section, figure, or table
it covers, so a failure points at the sentence that has to change.

Run with `make test`, or directly:

```bash
python3 tests/test_paper_numbers.py
```

This is the complement to `make validate`, which checks the regenerated tables against the frozen
ones rather than against the paper's prose.
