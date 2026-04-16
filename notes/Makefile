NOTES  = ic_sampling_review lpt_review fft_review
PYTHON = conda run -n cosmo python

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
    OPEN = evince
else ifeq ($(UNAME), Darwin)
    OPEN = open
endif

TIDY = rm -f *.aux *.log *.blg *.out *.toc

FIGURES = figures/box_window.pdf figures/tophat_window.pdf figures/pgrid_comparison.pdf \
          figures/pancake.pdf figures/growth.pdf figures/dft_pairs.pdf figures/cft_pairs.pdf

all : figures notes

# --- Figures ---
figures : $(FIGURES)

figures/box_window.pdf : figures/plot_box_window.py
	cd figures && $(PYTHON) plot_box_window.py

figures/tophat_window.pdf : figures/plot_tophat_window.py
	cd figures && $(PYTHON) plot_tophat_window.py

figures/pgrid_comparison.pdf : figures/plot_pgrid.py ../data/class_pk_z45_pk.dat
	cd figures && $(PYTHON) plot_pgrid.py

figures/pancake.pdf : figures/plot_pancake.py
	cd figures && $(PYTHON) plot_pancake.py

figures/growth.pdf : figures/plot_growth.py
	cd figures && $(PYTHON) plot_growth.py

figures/dft_pairs.pdf : figures/plot_dft_pairs.py
	cd figures && $(PYTHON) plot_dft_pairs.py

figures/cft_pairs.pdf : figures/plot_cft_pairs.py
	cd figures && $(PYTHON) plot_cft_pairs.py

# --- LaTeX ---
notes : $(FIGURES) $(addsuffix .pdf, $(NOTES))

%.pdf : %.tex
	pdflatex -draftmode -synctex=1 $<
	-bibtex $(basename $<)
	pdflatex -draftmode -synctex=1 $<
	pdflatex $<
	$(OPEN) $@ &
	$(TIDY)

clean :
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.synctex.gz
	rm -f figures/*.png $(FIGURES) $(NOTES).pdf
