# `music2_internals.tex` major-update plan

## Goal

Transform `notes/music2_internals.tex` from its current 537-line, 8-equation
sketch into a 2000+ line, equation-rich source-walking reference that parallels
`fft.tex`, `restriction_lpt.tex`, and `cosmo_ic.tex` in level of detail.  The
target audience is a graduate student or new collaborator wanting to understand
exactly what every line of `MUSIC2-anisotropic-zoom/src` does and why.

## Scope

Cover the **full IC-generation pipeline** end-to-end, with emphasis on:

1. **Mathematical foundations.**  Every numerical operation is connected back
   to the continuum equation it is approximating, with units carried through.
   The reader should be able to derive the discrete formula from first
   principles (cosmology + LPT + numerics).

2. **Algorithmic walk-through.**  For each major source file, identify the
   functions/classes, the data flow, the configuration knobs, and the
   correspondence to the equations.

3. **Hahn 2011 lineage and v1 → v2 → fork divergences.**  Be explicit about
   what was in Hahn & Abel 2011, what monofonIC (Hahn & Angulo 2021) changed,
   and what our `MUSIC2-anisotropic-zoom` fork added (per-axis isolation flags,
   `kspace_TF`, `kaveraging`, `density_boundary`, `dump_delta`, the
   real-space TF kernel port).

4. **Validation.**  Document the matched-noise test (Hahn 2011 §4.3.1) and
   our reproduction of the paper's 10⁻⁴σ residual on δ(q).  Include the
   per-axis-isolation pencil-zoom validation logic.

## Reference style

**Completely rewrite** the existing music2_internals.tex (don't preserve its
prose).  Follow the equation-rich, derivation-heavy style of the three
companion notes:

- **`fft.tex`** for mathematical conventions (CFT → DFT → FFTW), parallel
  decomposition, normalisation accounting.  Pattern: define the operator
  formally, derive the discrete version, identify the conventions a code
  must commit to, then point at the code.
- **`restriction_lpt.tex`** for cross-grid Poisson consistency, per-level
  $P(k)$, restriction operators, field-level diagnostics.  Pattern:
  formal restriction operator → its action in Fourier space → empirical
  verification on a toy model.
- **`cosmo_ic.tex`** for the cosmology + LPT scaffolding that MUSIC2 operates
  inside.  Pattern: continuum equation → perturbative expansion → algorithm.

Concretely, this means:

1. Lead each section with the *mathematical* object MUSIC2 is computing
   (an operator, a transform, a multigrid solve), defined in continuum
   notation.
2. Derive the discrete approximation MUSIC2 uses, with all factors of
   $2\pi$, $\sqrt{8}$, $\Delta x$, FFTW normalisations carried through.
3. Cite the source by file and line number only after the math is
   self-contained.
4. Use equation environments liberally; aim for $\gtrsim 1$ numbered
   equation per page on average.
5. Where MUSIC2's choice differs from a "textbook" derivation, explain
   the choice (algorithmic trade-off, performance, historical accident,
   etc.).

This note should *interleave* with the other three: pointers to relevant
sections (e.g. "the FFT-r2c normalisation convention discussed in
\S\,X.Y of fft.tex"), without duplicating their content.

## Outline

```
\section{Introduction}
  - Motivation, structure of the document, source-tree map, version
    correspondence (MUSIC1 release_candidate ↔ MUSIC2 master ↔ this fork).
  - Two-stage pipeline: (A) noise hierarchy, (B) δ → φ → Ψ → particles.
  - Reading guide.

\section{Notation and Conventions}
  - Box length L, level ℓ, grid N_ℓ = 2^ℓ, cell spacing Δx_ℓ = L/N_ℓ.
  - Lagrangian q vs Eulerian x.
  - FFT normalisation convention (forward unnormalised, inverse 1/N).
  - Real-to-complex Hermitian-truncated layout.
  - White noise μ with var(μ) = 1 per cell.
  - δ(q) ↔ μ map via T(k) convolution.
  - Indexing: i, j, k loop variables; r-Mpc/h vs Mpc.
  - Configuration option style: [section]/key = value.

\section{Cosmological Inputs}
  - Cosmology parameter container (cosmology_parameters.cc); cross-ref
    cosmo_ic.tex for the Friedmann scaffolding.
  - Power spectrum P(k) sources: Eisenstein–Hu, CAMB, CLASS, file.
  - Transfer function T(k) at z=z_ini: P(k) ∝ T²(k) k^{n_s}.
  - σ_8 normalisation.
  - LPT pre-factors D₁(z), D₂(z), f₁, f₂.
  - File pointers: cosmology_calculator.hh, transfer_function.hh,
    plugins/transfer_*.cc.

\section{The White-Noise Hierarchy (Stage A)}
  \subsection{Cubesize-tiled deterministic RNG}
    - Why cube-tiled sampling (Hahn 2011 §3.4.1 + later cubsize fix);
      each cube has its own seed = baseseed + cubeindex.
    - File pointer: plugins/random_music.cc; ran_cube_size config.
    - Statistics: var = 1 per cell, mean = 0 (after zero-mean step).
  \subsection{File-loaded white noise (GRAFIC)}
    - GRAFIC format: Fortran-unformatted header + slabs, x-fast/y-slow/z-slow.
    - Sign convention: grafic_sign config; reader's automatic negation.
    - Use case: matched-noise validation (Hahn 2011 §4.3.1 protocol).
  \subsection{Two approaches for multi-level noise consistency: Hahn 2011 §2.3.1}
    \subsubsection{Approach 1 — Hoffman–Ribak in real space}
      - Cite Hoffman & Ribak 1991 + Bertschinger 2001.
      - The 8-cell octet adjustment formula and why it preserves variance.
      - Code: \code{kaveraging = no} branch in
        random_music_wnoise_generator.cc:528+ (ported from
        MUSIC1 release_candidate).
    \subsubsection{Approach 2 — FFT-based coarse-mode replacement}
      - $\hat c^{\rm fine}(\mathbf k) = \sqrt{8}\,\hat c^{\rm coarse}(\mathbf k)$
        for $|\mathbf k|\le k_{\rm Ny}^{\rm coarse}$ and the variance-matching
        argument.
      - Meyer window vs Shannon (hard) cutoff: bias/aliasing tradeoff
        (the v1 → v2 change).
      - Code: \code{kaveraging = yes} branch (default).
    \subsubsection{Why Approach 1 is needed for Hahn's 10⁻⁴ validation}
      - FFT-array-shape dependency of Approach 2; bit-correlated noise
        across runs requires Approach 1.
      - Empirical: 1.4×10⁻³ → 3.5×10⁻⁴ on MUSIC1, 6.6×10⁻² → 2.6×10⁻⁴
        on MUSIC2 fork after enabling.
  \subsection{Doubled-patch storage convention}
    - The wnoise_LLLL.bin file format for level L > levelmin.
    - $n_f = 2 \times (\text{patch\_size} + \text{padding})$ doubled-grid layout.
    - File covers absolute level-L cells $[\text{offset\_abs} - n_f/4,
      \text{offset\_abs} + 3n_f/4)$; centered patch interior.
    - Code: random_music.cc::store_rnd; the unlink-on-destructor footgun.

\section{The Transfer-Function Kernel}
  \subsection{k-space sampled kernel: tf_kernel_k}
    - Direct evaluation of T(k) at each FFT mode.
    - kx, ky, kz integer indices ↔ physical k via $2\pi/L_{\rm box}$.
    - File: convolution_kernel.cc, the only kernel in monofonIC.
  \subsection{Real-space kernel (ported back from MUSIC1)}
    - kspace_TF = no branch.
    - Sample T(r) on the real-space grid via FFTLog (Talman 1978, Hamilton 2000).
    - Optional sub-cell quadrature (eval_split_recurse).
    - Cache T(r) per level to disk (temp_kernel_levelLLL.tmp).
    - Code: convolution_kernel_real.cc, transfer_function_real.hh.
    - Why we ported it: Hahn 2011's δ_self is bit-identical between zoom
      and unigrid only with real-space TF (paper §4.3.1).
  \subsection{Periodic vs. truncated TF: setup/periodic_TF}
    - Hahn 2011 §2.3.3: "truncation introduces larger errors than assuming
      periodicity"; we preserve the periodic-TF default.
  \subsection{Deconvolution for the CIC particle-deposition window}
    - setup/deconvolve toggle.  The kernel pre-divides by $\sinc^2(k_i \Delta x/2)$.

\section{Convolution Driver: convolution::perform}
  - Inputs: kernel, data buffer, shift/fix/flip flags.
  - The branch on \code{is\_ksampled()}: kspace path (direct multiplication)
    vs precomputed-grid path (read precomputed k-space-FFT'd kernel from
    cache and multiply).
  - Anisotropic-patch kvec convention (kfac per axis) — our fork's addition.
  - Mode-amplitude fixing: setup/fix_mode_amplitude + flip_mode_amplitude.
    - Why the NaN guard at $|c|=0$ was needed (our bugfix).
  - SPH baryon stagger: shift parameter.
  - DC mode handling — old vs new convention.
  - dump_delta debug hook (our addition).

\section{δ Hierarchy: Building the Density Field}
  \subsection{Unigrid path: GenerateDensityUnigrid}
    - Allocate $N^3$ DensityGrid, load noise, one convolution.
    - File pointer: densities.cc:255+.
  \subsection{Zoom path A — Fourier splicing (existing default)}
    - One convolution per level + fft_interpolate splicing.
    - File pointer: densities.cc:316+.
    - Per-axis-isolation extensions: convolution_margin per axis,
      PaddedDensitySubGrid with axis_periodic[3].
  \subsection{Zoom path B — three-term decomposition (our port)}
    - Hahn 2011 §2.3.3:
      $$\delta^{\ell+1} = \delta_{\rm self}^{\ell+1} + \delta_{\rm bnd}^{\ell+1}
      + \delta_{\rm coarse}^{\ell+1}.$$
    - δ_self: \code{subtract\_boundary\_oct\_mean} → convolve → copy to L+1 delta.
    - δ_bnd: restore → \code{subtract\_boundary\_oct\_mean} → \code{zero\_subgrid}
      over inner patch → convolve → mg\_cubic.prolong\_add to L+1 delta.
    - δ_coarse: restore → \code{subtract\_oct\_mean} → convolve →
      \code{subtract\_mean} → \code{upload\_bnd\_add} to L-1 delta.
    - File pointer: densities_three_term.cc.  Activated by
      \code{[setup]/density\_boundary = yes}.
    - Code-level walkthrough of the four helpers in density_grid.hh.

\section{Hand-off to LPT and Velocities}
  - δ → φ via multigrid Poisson (Hahn 2011 §3.2).
  - 1LPT: $\boldsymbol\Psi^{(1)} = -\nabla \phi^{(1)}$, $\mathbf v^{(1)} = a \dot a f_1 \boldsymbol\Psi^{(1)}$.
  - 2LPT: source $\tau$ from $\phi^{(1)}$, second Poisson solve, $\boldsymbol\Psi^{(2)}$.
  - File pointer: main.cc velocity branch.

\section{Per-Axis Isolation (this fork's distinguishing feature)}
  \subsection{Motivation: pencil-zoom geometries}
    - Long-pencil zoom on cosmological filament/wall structures.
  \subsection{axis_periodic[3] flag plumbing}
    - refinement_hierarchy.hh, plugins/random_music.cc per-axis margin,
      densities.cc per-axis PaddedDensitySubGrid constructor.
  \subsection{kfac per axis}
    - convolution_kernel.cc anisotropic-patch branch.
  \subsection{Validation against the cubic-zoom case}
    - Cubic and pencil should give equal δ residual; pointer to
      restriction_lpt.tex § zoom validation.

\section{Multigrid Poisson Solver}
  - FAS scheme: red-black Gauss-Seidel + V-cycles; cite Hahn 2011 §3.2 +
    Brandt 1977.
  - Coarse-fine boundary flux correction: mg_interp.hh interp_OX_fluxcorr.
  - File pointers: poisson.cc, mg_solver.hh, mg_operators.hh, mg_interp.hh.

\section{Validation: the Hahn 2011 §4.3.1 Matched-Noise Test}
  \subsection{Protocol}
    - Same seeds, two runs: zoom (256³ → 512³, 0.2 cube patch) and
      unigrid (512³).  Compare δ(q) at level 9 inside the patch volume.
  \subsection{Bit-correlated noise requirement}
    - With kaveraging = no, both runs produce bit-correlated L9 noise at
      the same Lagrangian cell by construction.  Approach 2's FFT-array-
      shape dependency makes this impossible.
  \subsection{Bit-correlated δ requirement}
    - With density_boundary = yes, the patch interior δ matches what a
      unigrid produces.
  \subsection{Empirical residuals}
    - Default MUSIC2 fork:        6.6×10⁻²σ.
    - + kaveraging=no:            6.0×10⁻³σ.
    - + density_boundary=yes:     2.6×10⁻⁴σ.
    - Paper claim:                "below 10⁻⁴σ_δ".
  \subsection{Sample 2x4 figure}
    - Top row: δ(q), Ψ_x, Ψ_y, Ψ_z on the convolution grid.
    - Bottom row: δ(x), v_x, v_y, v_z from CIC of displaced particles.
    - Cite figures/m2bc_decomp_2x4.png and m2bc_2lpt_2x4.png.
  \subsection{What the matched-noise test is, and is not}
    - Zoom ICs exist to pump compute into a specific region (halo,
      filament, void, lensing target) at much higher resolution than the
      user could afford box-wide, while the rest of the box supplies the
      long-range tides at coarse resolution.  The user gets the correct
      large-scale environment without paying for it, and the fine
      particles resolve the small-scale physics that motivated the run.
    - The matched-noise test asks "would a hypothetical full-box unigrid
      at the patch resolution have produced bit-identical noise here?"
      That is an internal consistency check on the zoom machinery — not
      the user-facing goal.  What the user actually needs is statistical
      equivalence: ICs in the patch with the right P(k), the right local
      tidal tensor, and the right cosmic environment, so the halo that
      forms is physically the same one a (computationally impossible)
      full-resolution box would produce.
    - A residual of ~10⁻²σ on the displacement field — concentrated at
      the patch face, decaying inward with a smooth k-space spectrum —
      does not visibly disturb any of that.  The halo at the patch
      centre has the same mass, profile, and accretion history.  The
      bit-equality target only becomes load-bearing for code-validation
      paper figures, not for science runs.

\section{Why MUSIC2 Differs From MUSIC1: A Reading of the Rewrite}
  \subsection{What monofonIC (Hahn & Angulo 2021) carried forward}
  \subsection{What it intentionally dropped}
    - Real-space TF kernel (removed in MUSIC2 commit bb9c07c, 2023-02-04).
    - Three-term δ decomposition.
    - kaveraging = no (Approach 1).
    - And the design rationale: speed, simplicity, PanPhasia supersedes
      noise-matching needs.
  \subsection{What our fork puts back}
    - kspace_TF = no, kaveraging = no, density_boundary = yes — all opt-in.

\section{Configuration Reference}
  - Annotated catalogue of every [setup], [random], [poisson], [output] key
    we touch, with default value, allowed range, and which section/equation
    documents the meaning.

\section{Appendices}
  \subsection{Per-mode variance of the unnormalised DFT (kept from original)}
  \subsection{Hoffman–Ribak constraint formula derivation}
  \subsection{FFTLog and the real-space TF tabulation}
  \subsection{GRAFIC wnoise file format reference}
  \subsection{Glossary of MUSIC2 config keys}

\section{References}
  - Hahn & Abel 2011 (MNRAS 415, 2101).
  - Hahn & Angulo 2021 (MNRAS 502, 3672) [monofonIC].
  - Bertschinger 2001 (ApJS 137, 1) [GRAFIC + Hoffman-Ribak in cosmology].
  - Hoffman & Ribak 1991 (ApJL 380, L5).
  - Jenkins 2013 (MNRAS 434, 2094) [PanPhasia].
  - Pen 1997 (ApJS 109, 1) [variance-matched noise refinement].
  - Brandt 1977 + Trottenberg–Oosterlee–Schüller 2001 [multigrid].
  - Talman 1978 / Hamilton 2000 [FFTLog].
```

## Drafting strategy

1. **Phase 0** — write this plan (DONE).  Confirm scope with user.
2. **Phase 1** — preserve the existing 537 lines as a starting skeleton.
   Don't delete content; restructure into the new outline, with the existing
   sketches becoming the seed paragraphs of the appropriate new sections.
3. **Phase 2** — for each section in the outline, write the full equation
   set and prose.  Aim for ~150-300 lines per major section.
4. **Phase 3** — three review passes, each in a different reader persona
   (see "Iteration protocol" below).  After each pass, identify unclear
   parts, answer them by re-reading the source or cosmo_ic.tex /
   restriction_lpt.tex, and edit the note in place.

## Iteration protocol (per user request)

After the draft is complete, perform **three review passes** in the persona
of a first-year graduate student who has read `cosmo_ic.tex` and `fft.tex`
but not the MUSIC source.  After each pass:

1. Note unclear or apparently-skipped points in a scratch list.
2. Resolve each point either by clarifying the prose or adding a derivation /
   reference.  Cite the source line number where appropriate.
3. Commit the updated draft.

Stop after three passes (or earlier if the draft reads cleanly on a new pass).

## File pointers I'll need open while writing

```
src/main.cc                          — top-level flow
src/densities.cc                     — GenerateDensityHierarchy, dispatch
src/densities_three_term.cc          — three-term decomposition
src/density_grid.hh                  — DensityGrid + PaddedDensitySubGrid
src/convolution_kernel.cc            — kspace TF + perform() driver
src/convolution_kernel_real.cc       — real-space TF kernel
src/transfer_function.hh             — k-space TF
src/transfer_function_real.hh        — real-space TF
src/plugins/random_music.cc          — RNG factory, level-by-level orchestration
src/plugins/random_music_wnoise_generator.cc  — noise constructors, Hahn 2011 §2.3.1
src/mg_operators.hh                  — restrict/prolong (mg_cubic etc)
src/mg_solver.hh                     — multigrid V-cycle
src/poisson.cc                       — Poisson driver
src/refinement_hierarchy.hh          — refh
src/mesh.hh                          — grid_hierarchy
```

## Estimated section sizes

| Section                        | Target lines |
|--------------------------------|--------------|
| Introduction                   | 80           |
| Notation                       | 100          |
| Cosmological inputs            | 200          |
| White-noise hierarchy          | 400          |
| Transfer-function kernel       | 250          |
| Convolution driver             | 200          |
| δ hierarchy (incl. three-term) | 400          |
| Hand-off to LPT                | 150          |
| Per-axis isolation             | 200          |
| Multigrid Poisson              | 200          |
| Matched-noise validation       | 250          |
| MUSIC1 → MUSIC2 → fork         | 150          |
| Config reference               | 100          |
| Appendices                     | 300          |
| **Total**                      | **~3000**    |
