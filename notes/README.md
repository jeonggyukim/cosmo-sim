# Notes

LaTeX write-ups on topics related to cosmological simulations and IC generation.

## Quick start

Build all PDFs (and regenerate figures as needed):

```bash
cd notes
make
```

This compiles `cosmo_ic.pdf`, `ic_sampling.pdf`, and
`restriction_lpt.pdf` and opens them automatically.

## Files

(Source `.tex` files live alongside; readers should open the
compiled `.pdf`, built by running `make` in this directory.)

| Note | Title | Contents |
|------|-------|----------|
| `cosmo_stat.pdf` | Statistical Foundations for Cosmological Fields | Why cosmology needs random fields (one universe, many possible ones), ensemble averages, statistical homogeneity and isotropy, ergodicity, ξ(r) as a covariance, Fourier conventions, P(k) and the variance budget, covariance as the unifying object, Gaussian random fields and Wick's theorem, generating a GRF on a computer (white noise → colouring), estimating ξ and P from one realisation, cosmic variance. Renamed from `foundations.pdf`. |
| `cosmo_ic.pdf` | Cosmological Initial Conditions: Theory and Practice | FLRW background, fluid equations, LPT (ZA, 2LPT), IC generation, P(k), ξ(r), ψ(r), starting redshifts; §11 MUSIC2 / monofonIC implementation details. Notation table at the start (§1). |
| `ic_sampling.pdf` | Real-Space vs Fourier-Space IC Sampling | Pen 1997, Sirko 2005, Hahn 2011: P-sampled vs ξ-sampled ICs, box window errors |
| `restriction_lpt.pdf` | Restriction of a Density Field and the 1LPT Displacement | Bin-averaging restriction (cell window + aliasing), effect on δ / Ψ⁽¹⁾ / deformation tensor / S⁽²⁾, why restricting the displacement potential is Poisson-inconsistent, application to zoom-in IC generation |
| `music2_internals.pdf` | MUSIC2 Source Walkthrough: How δ and the Noise Hierarchy Are Built | File/line walkthrough of `cosmo-sims/MUSIC2` (commit `967651b`): unigrid δ construction (k-space multiplier, FFT bookkeeping, DC mode, baryon stagger), zoom v2 noise hierarchy (sqrt8 variance match, half-cell phase, Meyer blending), zoom δ assembly. Source-walking counterpart to `cosmo_ic.pdf` §11. |
| `seed_selection.pdf` | Selecting a Random Seed on the Correlation Function of a Subvolume | Why adopting the seed whose subvolume ξ(r) best matches linear theory replaces the intended ensemble by a conditional one. Exact two-dice example, conditional moments of a Gaussian pair, ξ as a sum of squared mode amplitudes (~10 of them for a 125×125×1000 Mpc/h subvolume), integral-constraint displacement of the target. Measured on 600 realisations at 256³: scatter falls to 0.38–0.64 within the matched range and 0.67–0.94 outside it, means at unmatched separations move up to 2.8 standard errors, no tolerance threshold exists (worst-fifth rejection already removes 22%), the closest of 600 seeds still misses by 0.19σ, and a 10% estimator bias moves the measurement by only 0.068σ. Recommendations: compare against ⟨ξ̂⟩ not raw theory, report the band, fixed-amplitude or constrained ICs, reject on subvolume mean density alone (removes no scatter). Journal register, unlike the other notes here — its readers are collaborators evaluating an argument. |
| `xi_estimators.pdf` | The Grid Correlation Function: the Direct Lag Sum and the Padded FFT Are the Same Estimator | The grid estimator as Landy–Szalay in the infinite-random limit (RR evaluated analytically) and in Slepian & Eisenstein's general N = D−R form, where LS is itself a binned convolution; Wiener–Khinchin as a finite identity from DFT orthogonality; circular vs linear autocorrelation and the padding condition d ≤ P−N (pad by the largest separation wanted, not by the region size); the separable pair count N or N−d per axis; mixed per-axis boundaries for a pencil beam; rounding; cost scaling and when each method wins; what the Fourier route concedes (gridding smoothing, Cartesian-vs-spherical binning, misnormalisation if the denominator does not match); the integral constraint; what a two-method comparison does not validate. |

The dark-matter physics notes (`sidm`, `fdm`, `dm_chemistry`, `qft`,
`classical_mechanics`) moved to the `ai-notes` repository on 2026-07-28, where
they became `sidm.tex`, `sidm_3d.tex`, `kinetic_theory.tex`, `fdm.tex`,
`atomic_dm.tex`, `qft.tex` and `classical_mechanics.tex`. They had no
dependency on this repository; what remains here is the IC-generation and
statistics material, three figures of which are built from pipeline output.

## Reading order

Suggested progression for self-study:

1. **`cosmo_ic.pdf`** — the spine: cosmology background → linear theory → LPT → IC generation pipeline → MUSIC2 internals. Start here if you have a Fourier-analysis background.
2. **`ic_sampling.pdf`** — focused deep dive into how white noise is generated in IC codes (P-sampled vs ξ-sampled, box window). Reads naturally after `cosmo_ic.pdf` but is largely self-contained.
3. **`restriction_lpt.pdf`** — focused application: what restriction (block-averaging) does to fields and why bin-averaging shouldn't be used to construct zoom-in lo-res ICs. Builds on the LPT machinery from `cosmo_ic.pdf`.

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
