NOTES  = ic_sampling_review
PYTHON = conda run -n cosmo python

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
    OPEN = evince
else ifeq ($(UNAME), Darwin)
    OPEN = open
endif

TIDY = rm -f *.aux *.log *.blg *.out *.toc

FIGURES = box_window.pdf tophat_window.pdf pgrid_comparison.pdf

all : figures notes

# --- Figures ---
figures : $(FIGURES)

box_window.pdf : plot_box_window.py
	$(PYTHON) $<

tophat_window.pdf : plot_tophat_window.py
	$(PYTHON) $<

pgrid_comparison.pdf : plot_pgrid.py ../data/class_pk_z45_pk.dat
	$(PYTHON) $<

# --- LaTeX ---
notes : $(FIGURES) $(NOTES).tex
	pdflatex -draftmode -synctex=1 $(NOTES).tex
	pdflatex $(NOTES).tex
	$(OPEN) $(NOTES).pdf &
	$(TIDY)

clean :
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.synctex.gz
	rm -f *.png $(FIGURES) $(NOTES).pdf
