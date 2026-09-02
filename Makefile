.PHONY: analysis validate test figures all clean

PY := $(shell test -x .venv/bin/python3 && echo .venv/bin/python3 || echo python3)

# Regenerate every derived table from data/ into analysis_out/. Idempotent, CPU-only, no GPU,
# no dataset access, no network.
analysis:
	mkdir -p analysis_out/claim1 analysis_out/claim2
	$(PY) scripts/claim1/build_ranking_comparison.py
	$(PY) scripts/claim1/build_pairwise_concordance.py
	$(PY) scripts/claim1/build_force_scaling.py
	$(PY) scripts/claim1/build_dense_grid_pairwise_gap.py
	$(PY) scripts/claim1/build_dense_grid_ranking_comparison.py
	$(PY) scripts/claim1/build_frontier_points.py
	$(PY) scripts/claim2/build_dose_response.py
	$(PY) scripts/claim2/build_compensation.py
	$(PY) scripts/claim2/build_same_width_control_B.py
	$(PY) scripts/claim2/build_run_replication_control_C.py
	$(PY) scripts/claim2/build_frontier_ell4_degree_balanced.py
	$(PY) scripts/claim2/build_seed_replication_result.py
	$(PY) scripts/claim2/build_depth_localization.py
	$(PY) scripts/claim2/build_lmax24_frontier.py
	$(PY) scripts/claim2/build_ood_curves_and_contrasts.py
	$(PY) scripts/claim2/build_ood_absolute_effect_robustness.py
	$(PY) scripts/claim2/build_ood_domain_decomposition.py

# Check every regenerated table against the frozen table it reproduces.
validate: analysis
	$(PY) scripts/validate.py

# Check every number the paper reports against the regenerated tables.
test: analysis
	$(PY) tests/test_paper_numbers.py

# Render the two paper figures from the regenerated tables. Plotting only: no scientific
# quantity is computed here, which is why this is not a dependency of analysis or validate.
figures: analysis
	mkdir -p figures
	cd scripts/figures && ../../$(PY) figure1.py
	cd scripts/figures && ../../$(PY) figure2.py

all: validate test figures

clean:
	rm -rf analysis_out/claim1 analysis_out/claim2
