
FIGDIR := ../figures

COLLECTIVE_FIGURES := collectives_scaleB collectives_scaleP
RAND_EVAL_FIGURES := compare-coo compare-coo-scalennz comm-accuracy
SUITESPARSE_FIGURES := suite_sparse_manifest suite_sparse_gpu_comparison suite_sparse_comm_model  suite_sparse_gpu_eval 
MATRIX_SPACE_FIGURES := matrix-space-gpu matrix-space-speedup matrix-space-wse matrix-space-wse-mincomm matrix-space-wse-tp

FIGURES := $(COLLECTIVE_FIGURES) $(SUITESPARSE_FIGURES) $(RAND_EVAL_FIGURES) $(MATRIX_SPACE_FIGURES)
TARGETS := $(addprefix $(FIGDIR)/,$(addsuffix .pdf,$(FIGURES))) $(FIGDIR)/suite_sparse_matrices.tex

.PHONY: all clean
all: $(TARGETS)

$(FIGDIR)/%.pdf: %.csv plot_%.py | $(FIGDIR)
	python3 plot_$*.py

$(FIGDIR)/%.pdf: plot_%.py | $(FIGDIR)
	python3 plot_$*.py

$(FIGDIR)/suite_sparse_manifest.pdf: 080626manifest_full-coprimes.csv
	python3 plot_suite-sparse-manifest.py 080626manifest_full-coprimes.csv

$(FIGDIR)/suite_sparse_gpu_comparison.pdf: ss-results/073026csr-manifest.csv ss-results/080326coo-512cyclic.csv plot_suite_sparse_comparison.py
	python3 plot_suite_sparse_comparison.py

$(FIGDIR)/suite_sparse_comm_model.pdf: ss-results/080326coo-512cyclic.csv plot_suite_sparse_comm_model.py
	python3 plot_suite_sparse_comm_model.py

$(FIGDIR)/compare-coo.pdf: plot_compare-coo.py random-vary-matrix-nrows.csv rand-scale-rows-results080326-coo.csv
	python3 plot_compare-coo.py

$(FIGDIR)/compare-coo-scalennz.pdf: plot_compare-coo-scalennz.py random-vary-nnz.csv rand-scale-nnz-results080326-coo.csv
	python3 plot_compare-coo-scalennz.py

$(FIGDIR)/suite_sparse_matrices.tex: ss-results/matrix-stats.csv plot_suite_sparse_matrices_totex.py
	python3 plot_suite_sparse_matrices_totex.py

$(FIGDIR):
	mkdir -p $(FIGDIR)

clean:
	rm -f $(TARGETS)
