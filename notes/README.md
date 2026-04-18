# Notes

LaTeX write-ups on topics related to cosmological simulations and IC generation.

## Files

| File | Title | Contents |
|------|-------|----------|
| `cosmo_ic.tex` | Cosmological Initial Conditions: Theory and Validation | LPT (ZA, 2LPT), IC generation, P(k), ξ(r), ψ(r), starting redshifts, P(k) estimation; §11 MUSIC2 / monofonIC implementation details |
| `ic_sampling.tex` | Real-Space vs Fourier-Space IC Sampling | Pen 1997, Sirko 2005, Hahn 2011: P-sampled vs ξ-sampled ICs, box window errors |
| `fft.tex` | Fourier Transforms in Practice | DFT, FFTW, multi-D, MPI (fftMPI/Tigris BlockFFT) |

## Building

```bash
cd notes
make            # compile all PDFs (opens automatically)
make figures    # regenerate figures only
make notes      # compile PDFs only
make clean      # remove aux files, figures, PDFs
```

## Disclaimer

> **These notes were compiled by an AI assistant (Claude, Anthropic) and have not been reviewed by a human domain expert.**
> They may contain errors in derivations, physical reasoning, or numerical estimates.
> Cross-check key equations against the primary literature (Dodelson 2021, Baumann lecture notes, Jeong 2010, Bernardeau et al. 2002, etc.) before using them in research.

Each note also carries this disclaimer on its first page.
