
FIGDIR := ../figures


FIGURES := collectives_scaleB collectives_scaleP spmv-random-uniform suite_sparse_matrices suite_sparse_gpu_eval 
TARGETS := $(addprefix $(FIGDIR)/,$(addsuffix .pdf,$(FIGURES))) $(FIGDIR)/suite_sparse_manifest.pdf $(FIGDIR)/suite_sparse_gpu_comparison.pdf

.PHONY: all clean
all: $(TARGETS)

$(FIGDIR)/%.pdf: %.csv plot_%.py | $(FIGDIR)
	python3 plot_$*.py

$(FIGDIR)/suite_sparse_manifest.pdf: suite-sparse-manifest073026.csv suite-sparse-results073026.csv plot_suite-sparse-manifest073026.py
	python3 plot_suite-sparse-manifest073026.py

$(FIGDIR)/suite_sparse_gpu_comparison.pdf: suite-sparse-manifest073026.csv suite-sparse-results073026.csv plot_suite_sparse_comparison.py
	python3 plot_suite_sparse_comparison.py

$(FIGDIR):
	mkdir -p $(FIGDIR)

clean:
	rm -f $(TARGETS)
