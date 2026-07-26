# Notes

LaTeX write-ups on topics related to cosmological simulations and IC generation.

## Quick start

Build all PDFs (and regenerate figures as needed):

```bash
cd notes
make
```

This compiles `fft.pdf`, `cosmo_ic.pdf`, `ic_sampling.pdf`, and
`restriction_lpt.pdf` and opens them automatically.

## Files

(Source `.tex` files live alongside; readers should open the
compiled `.pdf`, built by running `make` in this directory.)

| Note | Title | Contents |
|------|-------|----------|
| `fft.pdf` | Fourier Transforms in Practice | DFT, FFTW, multi-D, MPI (fftMPI/Tigris BlockFFT) |
| `cosmo_stat.pdf` | Statistical Foundations for Cosmological Fields | Why cosmology needs random fields (one universe, many possible ones), ensemble averages, statistical homogeneity and isotropy, ergodicity, ξ(r) as a covariance, Fourier conventions, P(k) and the variance budget, covariance as the unifying object, Gaussian random fields and Wick's theorem, generating a GRF on a computer (white noise → colouring), estimating ξ and P from one realisation, cosmic variance. Renamed from `foundations.pdf`. |
| `cosmo_ic.pdf` | Cosmological Initial Conditions: Theory and Practice | FLRW background, fluid equations, LPT (ZA, 2LPT), IC generation, P(k), ξ(r), ψ(r), starting redshifts; §11 MUSIC2 / monofonIC implementation details. Notation table at the start (§1). |
| `ic_sampling.pdf` | Real-Space vs Fourier-Space IC Sampling | Pen 1997, Sirko 2005, Hahn 2011: P-sampled vs ξ-sampled ICs, box window errors |
| `restriction_lpt.pdf` | Restriction of a Density Field and the 1LPT Displacement | Bin-averaging restriction (cell window + aliasing), effect on δ / Ψ⁽¹⁾ / deformation tensor / S⁽²⁾, why restricting the displacement potential is Poisson-inconsistent, application to zoom-in IC generation |
| `music2_internals.pdf` | MUSIC2 Source Walkthrough: How δ and the Noise Hierarchy Are Built | File/line walkthrough of `cosmo-sims/MUSIC2` (commit `967651b`): unigrid δ construction (k-space multiplier, FFT bookkeeping, DC mode, baryon stagger), zoom v2 noise hierarchy (sqrt8 variance match, half-cell phase, Meyer blending), zoom δ assembly. Source-walking counterpart to `cosmo_ic.pdf` §11. |
| `classical_mechanics.pdf` | Classical Mechanics for Simulation Work | Lagrangian recap, the Legendre transform and where the Hamiltonian comes from, Hamilton's equations, phase space (Γ vs μ), Poisson brackets, canonical transformations (three tests, and why `det J = 1` is insufficient for n > 1), Liouville's theorem, symplectic integrators (why leapfrog beats Runge–Kutta over 10⁶ orbits, the shadow Hamiltonian, where adaptive steps and velocity-dependent forces break the argument). Supplies the mechanics that `sidm.pdf` App. A and `qft.pdf` §3/§7 assume. |
| `qft.pdf` | Field Theory Foundations for Scalar Dark Matter | Expanded from the `fdm.pdf` QFT appendix: least action, relativistic notation, free scalar field, Noether's theorem, canonical quantisation, coherent states and the classical-wave limit (why FDM's n~10⁹⁵ occupation makes the classical treatment exact), axion potential and naturalness, curved-spacetime coupling, convention traps. |
| `sidm.pdf` | Self-Interacting Dark Matter | Cross sections and the mean free path, the cored and core-collapsed regimes, velocity dependence and the Yukawa dark force, the gravothermal fluid framework, halo phenomenology (isothermal cores, baryonic backreaction, subhalo core collapse and the diversity problem), numerical methods, observational constraints, connections to SMBH seeding and atomic/fuzzy DM. Appendix A is the full kinetic-theory chain — Hamiltonian N-body → Liouville → BBGKY → Boltzmann → moments → Chapman–Enskog — written so the step that fails in the intermediate-mean-free-path shell can be identified precisely. |
| `fdm.pdf` | Fuzzy Dark Matter and BECDM | Ultralight scalars and misalignment production, the non-relativistic reduction to Schrödinger–Poisson, linear theory (Jeans scale, quantum pressure), soliton cores, numerical methods, astrophysical signatures and constraints. The QFT primer appendix was expanded into `qft.pdf`. |
| `dm_chemistry.pdf` | Molecular Chemistry for Atomic Dark Matter | The atomic-DM model and how it rescales from Standard Model hydrogen, dark recombination and freeze-out, dark diffusion and dark acoustic oscillations, cooling and compact-object formation, the modified Recfast++ / DarkKROME codes, 3D cosmological simulations, observational hooks. Appendices: hydrogenic atom from first principles, and a cosmology primer aimed at ISM/star-formation readers. |

## Reading order

Suggested progression for self-study:

1. **`fft.pdf`** — foundational Fourier-analysis tools used by everything else (sampling, Nyquist, aliasing, convolution theorem). Skim if you already know this material.
2. **`cosmo_ic.pdf`** — the spine: cosmology background → linear theory → LPT → IC generation pipeline → MUSIC2 internals. Start here if you have a Fourier-analysis background.
3. **`ic_sampling.pdf`** — focused deep dive into how white noise is generated in IC codes (P-sampled vs ξ-sampled, box window). Reads naturally after `cosmo_ic.pdf` but is largely self-contained.
4. **`restriction_lpt.pdf`** — focused application: what restriction (block-averaging) does to fields and why bin-averaging shouldn't be used to construct zoom-in lo-res ICs. Builds on the LPT machinery from `cosmo_ic.pdf` and the Fourier tools from `fft.pdf`; cross-references both.

Within each note the sections are ordered linearly; `cosmo_ic.pdf` (the longest) opens with a notation table for readers who want to jump in mid-document.

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
