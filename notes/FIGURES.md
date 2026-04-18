# Figures

Index of figures generated for the LaTeX notes. All scripts live in `figures/`
and write their PDFs into the same directory; the `notes/Makefile` drives them.

## By note

### `cosmo_ic.tex`

| PDF | Script | Used in |
|-----|--------|---------|
| `growth.pdf` | `plot_growth.py` | §2.2 growth factor $D_+(z)$ and rate $f(z)$ |
| `pancake.pdf` | `plot_pancake.py` | §4.4 Zel'dovich pancake (density + trajectories) |

Inline TikZ: IC-generation flowchart at end of §6.

### `fft.tex`

| PDF | Script | Used in |
|-----|--------|---------|
| `cft_pairs.pdf` | `plot_cft_pairs.py` | §1 four illustrative continuous FT pairs |
| `dft_pairs.pdf` | `plot_dft_pairs.py` | §3 four DFT input/output pairs |
| `aliasing.pdf` | `plot_aliasing.py` | §3 aliasing from undersampling |

Inline TikZ: $N=8$ Cooley–Tukey butterfly flow graph in §2.1.

### `ic_sampling.tex`

| PDF | Script | Used in |
|-----|--------|---------|
| `box_window.pdf` | `plot_box_window.py` | §box window truncation in Fourier space |
| `tophat_window.pdf` | `plot_tophat_window.py` | §top-hat window for ξ-sampling |
| `pgrid_comparison.pdf` | `plot_pgrid.py` | §P-sampled vs ξ-sampled $P_{\rm grid}(k)$ (needs `data/class_pk_z2_pk.dat`) |

## Regeneration

```bash
cd notes
make figures     # rebuild all PDFs via the Makefile
```

Individual plots can be rerun directly:

```bash
cd notes/figures
conda run -n cosmo python plot_<name>.py
```
