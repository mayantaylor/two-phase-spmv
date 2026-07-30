
FIGDIR := ../figures

FIGURES := collectives_scaleB collectives_scaleP spmv-random-uniform suite_sparse_matrices suite_sparse_gpu_eval
TARGETS := $(addprefix $(FIGDIR)/,$(addsuffix .pdf,$(FIGURES)))

.PHONY: all clean
all: $(TARGETS)

$(FIGDIR)/%.pdf: %.csv plot_%.py | $(FIGDIR)
	python3 plot_$*.py

$(FIGDIR):
	mkdir -p $(FIGDIR)

clean:
	rm -f $(TARGETS)
