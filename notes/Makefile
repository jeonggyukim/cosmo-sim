NOTES  = ic_sampling cosmo_ic fft restriction_lpt
PYTHON = conda run -n cosmo python

# Absolute path to this Makefile's directory — pdflatex is always run from here
# so .log/.out/.aux files land in notes/ regardless of where make is invoked.
NOTESDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
    OPEN = evince
else ifeq ($(UNAME), Darwin)
    OPEN = open
endif

TIDY = rm -f *.aux *.log *.blg *.out *.toc

FIGURES = figures/box_window.pdf figures/tophat_window.pdf figures/pgrid_comparison.pdf \
          figures/pancake.pdf figures/growth.pdf figures/dft_pairs.pdf figures/cft_pairs.pdf \
          figures/aliasing.pdf \
          figures/restriction_density.pdf figures/restriction_psi.pdf \
          figures/restriction_deformation.pdf figures/restriction_phi.pdf \
          figures/restriction_phi_spectral.pdf \
          figures/restriction_cdm.pdf figures/restriction_cdm_truth.pdf \
          figures/window_compare.pdf \
          figures/alias_diagram.pdf figures/zoom_ic_comparison.pdf \
          figures/zoom_ic_pk_full.pdf

all : figures notes

# --- Figures ---
figures : $(FIGURES)

figures/box_window.pdf : figures/plot_box_window.py
	cd figures && $(PYTHON) plot_box_window.py

figures/tophat_window.pdf : figures/plot_tophat_window.py
	cd figures && $(PYTHON) plot_tophat_window.py

figures/pgrid_comparison.pdf : figures/plot_pgrid.py ../data/class_pk_z200_pk.dat
	cd figures && $(PYTHON) plot_pgrid.py

# Generate CLASS P(k) at z=45 from any existing MUSIC2-written CLASS ini.
# Picks the first conf/input_class_parameters_*.ini (they share cosmology);
# fails with a clear message if none exist (run the pipeline once first).
REPO_ROOT  := $(abspath $(NOTESDIR)/..)
# Prefer the in-tree CLASS built by ./build-music.sh (version-matched to MUSIC2);
# fall back to a `class` on $PATH so the notes can build without MUSIC2 present.
CLASS_BIN  := $(firstword $(wildcard $(REPO_ROOT)/music_build/_deps/class-build/class) $(shell command -v class))
CLASS_INI  := $(firstword $(wildcard $(REPO_ROOT)/conf/input_class_parameters_*.ini))

../data/class_pk_z200_pk.dat :
	@test -n "$(CLASS_INI)" || { echo "ERROR: no conf/input_class_parameters_*.ini found; run ./run_pipeline.sh once to produce one." >&2; exit 1; }
	@test -x "$(CLASS_BIN)"  || { echo "ERROR: no CLASS binary found (looked in music_build/_deps/class-build/ and \$$PATH); run ./build-music.sh or install CLASS." >&2; exit 1; }
	TMP=$$(mktemp /tmp/class_pk_XXXXXX) && mv "$$TMP" "$$TMP.ini" && TMP="$$TMP.ini" && \
	LC_ALL=C sed -e 's/^output =.*/output = mPk/' -e 's/^z_pk =.*/z_pk = 200/' \
	    -e '/^extra metric transfer functions/d' -e '/^gauge/d' \
	    $(CLASS_INI) > $$TMP && \
	echo "root = class_pk_z200_" >> $$TMP && \
	cd $(REPO_ROOT)/data && $(CLASS_BIN) $$TMP && \
	rm -f $$TMP

figures/pancake.pdf : figures/plot_pancake.py
	cd figures && $(PYTHON) plot_pancake.py

figures/growth.pdf : figures/plot_growth.py
	cd figures && $(PYTHON) plot_growth.py

figures/dft_pairs.pdf : figures/plot_dft_pairs.py
	cd figures && $(PYTHON) plot_dft_pairs.py

figures/cft_pairs.pdf : figures/plot_cft_pairs.py
	cd figures && $(PYTHON) plot_cft_pairs.py

figures/aliasing.pdf : figures/plot_aliasing.py
	cd figures && $(PYTHON) plot_aliasing.py

figures/restriction_density.pdf : figures/plot_restriction_density.py
	cd figures && $(PYTHON) plot_restriction_density.py

figures/restriction_psi.pdf : figures/plot_restriction_psi.py
	cd figures && $(PYTHON) plot_restriction_psi.py

figures/restriction_deformation.pdf : figures/plot_restriction_deformation.py
	cd figures && $(PYTHON) plot_restriction_deformation.py

figures/restriction_phi.pdf : figures/plot_restriction_phi.py
	cd figures && $(PYTHON) plot_restriction_phi.py

figures/restriction_phi_spectral.pdf : figures/plot_restriction_phi_spectral.py
	cd figures && $(PYTHON) plot_restriction_phi_spectral.py

figures/restriction_cdm.pdf : figures/plot_restriction_cdm.py ../data/class_pk_z200_pk.dat
	cd figures && $(PYTHON) plot_restriction_cdm.py

figures/restriction_cdm_truth.pdf : figures/plot_restriction_cdm_truth.py ../data/class_pk_z200_pk.dat
	cd figures && $(PYTHON) plot_restriction_cdm_truth.py

figures/window_compare.pdf : figures/plot_window_compare.py
	cd figures && $(PYTHON) plot_window_compare.py

figures/alias_diagram.pdf : figures/plot_alias_diagram.py
	cd figures && $(PYTHON) plot_alias_diagram.py

figures/zoom_ic_comparison.pdf : figures/plot_zoom_ic_comparison.py
	cd figures && $(PYTHON) plot_zoom_ic_comparison.py

figures/zoom_ic_pk_full.pdf : figures/plot_zoom_ic_pk_full.py
	cd figures && $(PYTHON) plot_zoom_ic_pk_full.py

# --- LaTeX ---
notes : $(FIGURES) $(addsuffix .pdf, $(NOTES))

%.pdf : %.tex
	cd $(NOTESDIR) && pdflatex -draftmode -synctex=1 $(notdir $<)
	cd $(NOTESDIR) && bibtex $(basename $(notdir $<)) || true
	cd $(NOTESDIR) && pdflatex -draftmode -synctex=1 $(notdir $<)
	cd $(NOTESDIR) && pdflatex $(notdir $<)
	$(OPEN) $(NOTESDIR)$(notdir $@) &
	cd $(NOTESDIR) && $(TIDY)

clean :
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.synctex.gz
	rm -f figures/*.png $(FIGURES) $(addsuffix .pdf, $(NOTES))
