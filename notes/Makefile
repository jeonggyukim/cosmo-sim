NOTES  = cosmo_stat ic_sampling cosmo_ic restriction_lpt music2_internals xi_estimators \
         ic_search
PYTHON = conda run -n cosmo python

# Absolute path to this Makefile's directory — pdflatex is always run from here
# so .log/.out/.aux files land in notes/ regardless of where make is invoked.
NOTESDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
    # $(1) = absolute PDF path
    define OPEN_PDF
	evince $(1) &
    endef
else ifeq ($(UNAME), Darwin)
    # Reload the PDF in Preview without stealing focus or spawning a new window.
    # If a document with the same path is already open, close+reopen it in place;
    # otherwise just open it. `activate` is omitted so Preview stays in the back.
    define OPEN_PDF
	osascript -e 'tell application "Preview"' \
	          -e 'set thePath to POSIX file "$(1)"' \
	          -e 'set theDocs to (every document whose path is POSIX path of thePath)' \
	          -e 'repeat with d in theDocs' \
	          -e 'close d saving no' \
	          -e 'end repeat' \
	          -e 'open thePath' \
	          -e 'end tell' >/dev/null
    endef
endif

TIDY = rm -f *.aux *.log *.blg *.out *.toc

# Figures live in subdirectories under figures/, grouped by parent .tex.
# Scripts that are shared by several .tex (growth, aliasing) stay at the top.
FIGURES = \
    figures/growth.pdf figures/aliasing.pdf \
    figures/ic_sampling/box_window.pdf \
    figures/ic_sampling/tophat_window.pdf \
    figures/ic_sampling/pgrid_comparison.pdf \
    figures/cosmo_ic/pancake.pdf \
    figures/cosmo_ic/shotnoise_order.pdf \
    figures/cosmo_ic/ngenic_seedtable.pdf \
    figures/cosmo_ic/mpi_fft_slabs.pdf \
    figures/fft/grf_exponential.pdf \
    figures/restriction/restriction_density.pdf \
    figures/restriction/restriction_psi.pdf \
    figures/restriction/restriction_deformation.pdf \
    figures/restriction/restriction_phi.pdf \
    figures/restriction/restriction_phi_spectral.pdf \
    figures/restriction/restriction_cdm.pdf \
    figures/restriction/restriction_cdm_truth.pdf \
    figures/restriction/window_compare.pdf \
    figures/restriction/alias_diagram.pdf \
    figures/restriction/zoom_ic_comparison.pdf \
    figures/restriction/zoom_ic_pk_full.pdf \
    figures/music2/meyer.pdf \
    figures/xi_estimators/xi_toy.pdf \
    figures/xi_estimators/xi_identity.pdf \


all : figures notes

# --- Figures ---
figures : $(FIGURES)

# Each plot script is invoked from its own directory so any relative paths
# inside the script (e.g. data lookups) stay sane and the resulting PDF
# lands next to the script.
figures/growth.pdf : figures/plot_growth.py
	cd figures && $(PYTHON) plot_growth.py

figures/aliasing.pdf : figures/plot_aliasing.py
	cd figures && $(PYTHON) plot_aliasing.py

figures/ic_sampling/box_window.pdf : figures/ic_sampling/plot_box_window.py
	cd figures/ic_sampling && $(PYTHON) plot_box_window.py

figures/ic_sampling/tophat_window.pdf : figures/ic_sampling/plot_tophat_window.py
	cd figures/ic_sampling && $(PYTHON) plot_tophat_window.py

figures/ic_sampling/pgrid_comparison.pdf : figures/ic_sampling/plot_pgrid.py ../data/class_pk_z200_pk.dat
	cd figures/ic_sampling && $(PYTHON) plot_pgrid.py

# Generate CLASS P(k) at z=45 from any existing MUSIC2-written CLASS ini.
# Picks the first conf/input_class_parameters_*.ini (they share cosmology);
# fails with a clear message if none exist (run the pipeline once first).
REPO_ROOT  := $(abspath $(NOTESDIR)/..)
# Prefer the in-tree CLASS built by tools/build-music.sh (version-matched to MUSIC2);
# fall back to a `class` on $PATH so the notes can build without MUSIC2 present.
CLASS_BIN  := $(firstword $(wildcard $(REPO_ROOT)/music_build/_deps/class-build/class) $(shell command -v class))
CLASS_INI  := $(firstword $(wildcard $(REPO_ROOT)/conf/input_class_parameters_*.ini))

../data/class_pk_z200_pk.dat :
	@test -n "$(CLASS_INI)" || { echo "ERROR: no conf/input_class_parameters_*.ini found; run run-pipeline once to produce one." >&2; exit 1; }
	@test -x "$(CLASS_BIN)"  || { echo "ERROR: no CLASS binary found (looked in music_build/_deps/class-build/ and \$$PATH); run tools/build-music.sh or install CLASS." >&2; exit 1; }
	TMP=$$(mktemp /tmp/class_pk_XXXXXX) && mv "$$TMP" "$$TMP.ini" && TMP="$$TMP.ini" && \
	LC_ALL=C sed -e 's/^output =.*/output = mPk/' -e 's/^z_pk =.*/z_pk = 200/' \
	    -e '/^extra metric transfer functions/d' -e '/^gauge/d' \
	    $(CLASS_INI) > $$TMP && \
	echo "root = class_pk_z200_" >> $$TMP && \
	cd $(REPO_ROOT)/data && $(CLASS_BIN) $$TMP && \
	rm -f $$TMP

figures/cosmo_ic/pancake.pdf : figures/cosmo_ic/plot_pancake.py
	cd figures/cosmo_ic && $(PYTHON) plot_pancake.py

figures/cosmo_ic/shotnoise_order.pdf : figures/cosmo_ic/plot_shotnoise_order.py
	cd figures/cosmo_ic && $(PYTHON) plot_shotnoise_order.py

figures/cosmo_ic/ngenic_seedtable.pdf : figures/cosmo_ic/plot_ngenic_seedtable.py
	cd figures/cosmo_ic && $(PYTHON) plot_ngenic_seedtable.py

figures/cosmo_ic/mpi_fft_slabs.pdf : figures/cosmo_ic/plot_mpi_fft_slabs.py
	cd figures/cosmo_ic && $(PYTHON) plot_mpi_fft_slabs.py

figures/fft/grf_exponential.pdf : figures/fft/plot_grf_exponential.py
	cd figures/fft && $(PYTHON) plot_grf_exponential.py

figures/restriction/restriction_density.pdf : figures/restriction/plot_restriction_density.py
	cd figures/restriction && $(PYTHON) plot_restriction_density.py

figures/restriction/restriction_psi.pdf : figures/restriction/plot_restriction_psi.py
	cd figures/restriction && $(PYTHON) plot_restriction_psi.py

figures/restriction/restriction_deformation.pdf : figures/restriction/plot_restriction_deformation.py
	cd figures/restriction && $(PYTHON) plot_restriction_deformation.py

figures/restriction/restriction_phi.pdf : figures/restriction/plot_restriction_phi.py
	cd figures/restriction && $(PYTHON) plot_restriction_phi.py

figures/restriction/restriction_phi_spectral.pdf : figures/restriction/plot_restriction_phi_spectral.py
	cd figures/restriction && $(PYTHON) plot_restriction_phi_spectral.py

figures/restriction/restriction_cdm.pdf : figures/restriction/plot_restriction_cdm.py ../data/class_pk_z200_pk.dat
	cd figures/restriction && $(PYTHON) plot_restriction_cdm.py

figures/restriction/restriction_cdm_truth.pdf : figures/restriction/plot_restriction_cdm_truth.py ../data/class_pk_z200_pk.dat
	cd figures/restriction && $(PYTHON) plot_restriction_cdm_truth.py

figures/restriction/window_compare.pdf : figures/restriction/plot_window_compare.py
	cd figures/restriction && $(PYTHON) plot_window_compare.py

figures/restriction/alias_diagram.pdf : figures/restriction/plot_alias_diagram.py
	cd figures/restriction && $(PYTHON) plot_alias_diagram.py

figures/restriction/zoom_ic_comparison.pdf : figures/restriction/plot_zoom_ic_comparison.py
	cd figures/restriction && $(PYTHON) plot_zoom_ic_comparison.py

figures/restriction/zoom_ic_pk_full.pdf : figures/restriction/plot_zoom_ic_pk_full.py
	cd figures/restriction && $(PYTHON) plot_zoom_ic_pk_full.py

figures/music2/meyer.pdf : figures/music2/plot_meyer.py
	cd figures/music2 && $(PYTHON) plot_meyer.py

figures/xi_estimators/xi_toy.pdf : figures/xi_estimators/plot_xi_toy.py
	cd figures/xi_estimators && $(PYTHON) plot_xi_toy.py

figures/xi_estimators/xi_identity.pdf : figures/xi_estimators/plot_xi_identity.py
	cd figures/xi_estimators && $(PYTHON) plot_xi_identity.py

# --- LaTeX ---
notes : $(FIGURES) $(addsuffix .pdf, $(NOTES))

%.pdf : %.tex
	cd $(NOTESDIR) && pdflatex -draftmode -synctex=1 $(notdir $<)
	cd $(NOTESDIR) && bibtex $(basename $(notdir $<)) || true
	cd $(NOTESDIR) && pdflatex -draftmode -synctex=1 $(notdir $<)
	cd $(NOTESDIR) && pdflatex $(notdir $<)
	$(call OPEN_PDF,$(NOTESDIR)$(notdir $@))
	cd $(NOTESDIR) && $(TIDY)

clean :
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.synctex.gz
	rm -f figures/*.png figures/*.pdf $(FIGURES) $(addsuffix .pdf, $(NOTES))
	rm -f figures/ic_sampling/*.png figures/cosmo_ic/*.png \
	      figures/fft/*.png figures/restriction/*.png figures/music2/*.png \
	      figures/ic_search/*.png \
	
