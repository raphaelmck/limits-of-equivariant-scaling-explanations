# figures/

Rendered by `make figures` from `scripts/figures/`, which read only from `analysis_out/` after
`make analysis` has regenerated and `make validate` has checked it. No scientific quantity is
computed in the plotting scripts; log axes and the exponentiation of the frozen power-law fit are
display transforms of already-validated numbers.

- `figure1.pdf` / `figure1.png` -- compute frontiers and power-law fits, force-NTK KRR
  learning-curve area, and the signed GemNet-OC/eSEN gap for both.
- `figure2.pdf` / `figure2.png` -- the block-9 ell=4 dose response, and log(L_2 / L_4) at the four
  matched-compute pairs on both evaluation populations.

`paper/FIGURE_PROVENANCE.md` maps each panel to the file it reads.
