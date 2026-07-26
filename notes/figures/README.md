# `notes/figures/` — plot scripts for the LaTeX notes

One Python script per figure used by the LaTeX write-ups in `notes/`.
Scripts are grouped into subdirectories matching their parent `.tex`;
the two scripts that several papers share (and the two orphan
exploratory scripts) stay at the top level.

Each script:
- runs from its own directory (the `notes/Makefile` does `cd` into the
  right subdir before invoking it);
- writes its output PDF next to itself (e.g. `restriction/plot_X.py`
  writes `restriction/X.pdf`);
- has its PDF/PNG output gitignored via `notes/figures/**/*.pdf`
  and `**/*.png`.

LaTeX finds the PDFs via a multi-path `\graphicspath{}` declared at the
top of each `.tex`, so `\includegraphics{X.pdf}` resolves regardless
of which subdirectory `X.pdf` lives in.

## Layout (script → parent .tex)

| subdir            | scripts                                                | parent .tex            |
|-------------------|--------------------------------------------------------|------------------------|
| `ic_sampling/`    | `plot_box_window`, `plot_tophat_window`, `plot_pgrid`  | `ic_sampling.tex`      |
| `cosmo_ic/`       | `plot_pancake`                                         | `cosmo_ic.tex`         |
| `fft/`            | `plot_grf_exponential` | shared with `cosmo_stat.tex` |
| `restriction/`    | `plot_restriction_{density,psi,deformation,phi,phi_spectral,cdm,cdm_truth}`, `plot_window_compare`, `plot_alias_diagram`, `plot_zoom_ic_{comparison,pk_full}` | `restriction_lpt.tex` |
| `music2/`         | `plot_meyer`                                           | `music2_internals.tex` |
| `figures/` (top)  | `plot_growth`, `plot_aliasing`                         | shared (used by 4 .tex each) |
| `figures/` (top)  | `plot_2lpt_cic_pk`, `plot_zoom_ic_pk`                  | orphan: exploratory, not in `notes/Makefile` |

## Adding a new figure

1. Drop the new `plot_X.py` into the subdir matching its parent `.tex`
   (or at the top level if it's shared / exploratory).
2. Save its PDF next to itself with `plt.savefig('X.pdf')`.
3. Add a target line in `notes/Makefile`:
   ```make
   figures/SUBDIR/X.pdf : figures/SUBDIR/plot_X.py
   	cd figures/SUBDIR && $(PYTHON) plot_X.py
   ```
   and append `figures/SUBDIR/X.pdf` to `FIGURES`.
4. `\includegraphics{X.pdf}` in the relevant `.tex` (no subdir prefix
   needed — the multi-path `\graphicspath{}` already covers all subdirs).
5. `make -C notes figures` to test; `make -C notes` to rebuild everything.

## Data dependencies

A few plot scripts read pipeline outputs from `../../../data/` (three
levels up from a subdir, or `../../data/` from the top level):

| script                                         | reads                                |
|------------------------------------------------|--------------------------------------|
| `ic_sampling/plot_pgrid.py`                    | `data/class_pk_z200_pk.dat`          |
| `restriction/plot_restriction_cdm.py`          | `data/class_pk_z200_pk.dat`          |
| `restriction/plot_restriction_cdm_truth.py`    | `data/class_pk_z200_pk.dat`          |

`notes/Makefile` knows about these dependencies; running `make
-C notes ../data/class_pk_z200_pk.dat` will generate the CLASS table
from any existing `conf/input_class_parameters_*.ini` (run the
pipeline once first).
