# Velocity power spectrum diagnostic and box-size sensitivity

Plan to add a velocity power-spectrum measurement tool to the pipeline,
and to demonstrate quantitatively why the velocity diagnostic is much
more sensitive to small box sizes than the density diagnostic.

Author: Claude (Anthropic), 2026-04-29.

## Motivation

The pipeline already validates ICs in two ways:
- **Density**: $P_\delta(k)$ via `scripts/compute_pk.py` (CIC + FFT + CIC
  window deconvolution) and $\xi(r)$ via `compute_xi` /
  `compute_xi_corrfunc`. Compared with linear theory in `plot_ic.py`.
- **Velocity**: $\psi(r) = \langle \mathbf v(\mathbf x) \cdot
  \mathbf v(\mathbf x+\mathbf r)\rangle$ via `compute_xi --vel`
  (CIC velocity grid auto-correlation). Compared with linear theory
  in `plot_ic.py`.

What's missing: a Fourier-space velocity diagnostic
$P_v(k) = (aHf)^2 P_\delta(k)/k^2$ that's directly comparable to the
density spectrum and visible on the same plot. This is useful for two
reasons:
1. Catches velocity-convention bugs (e.g. the $a\,v_{\rm pec}$ vs
   $v_{\rm pec}$ confusion we just patched) in Fourier-space form.
2. Lets us cleanly diagnose the **small-box velocity underestimation**
   problem — for small simulation boxes, the measured velocity
   correlation is much further below the linear-theory prediction
   than the corresponding density correlation. We want to understand
   and visualise this.

## Why velocity is more box-size-sensitive than density

In linear theory the irrotational peculiar-velocity field satisfies
$\mathbf v_{\rm pec}(\mathbf k) = i\,(aHf/k)\,\hat{\mathbf k}\,\delta(\mathbf k)$,
so the vector velocity power spectrum is
$$
  P_v(k) \;=\; \frac{(aHf)^2}{k^2}\,P_\delta(k).
$$

The **variance integrals** are dimensionally
$$
  \sigma_\delta^2 \;\propto\; \int_0^\infty k^2\,P_\delta(k)\,dk,
  \qquad
  \sigma_v^2     \;\propto\; \int_0^\infty P_\delta(k)\,dk.
$$
The $k^2$ factor in $\sigma_\delta^2$ makes density variance
small-scale-dominated. The absence of $k^2$ in $\sigma_v^2$ means
velocity variance is large-scale-dominated.

A finite simulation box of side $L$ truncates all modes below
$k_{\rm fund} = 2\pi/L$. The "missing" variance contribution is

| | density | velocity |
|---|---|---|
| missing variance | $\propto \int_0^{k_{\rm fund}} k^2 P_\delta(k)\,dk$ | $\propto \int_0^{k_{\rm fund}} P_\delta(k)\,dk$ |

For a $\Lambda$CDM-like spectrum at low $k$, $P_\delta(k) \propto k^{n_s}\sim k$.
Then:
- density: lost variance $\propto k_{\rm fund}^{n_s+3}\sim k_{\rm fund}^4$
  (small — strongly suppressed because of the $k^2$ measure)
- velocity: lost variance $\propto k_{\rm fund}^{n_s+1}\sim k_{\rm fund}^2$
  (much larger relative to total)

Halving the box doubles $k_{\rm fund}$, increasing the lost density
variance by $\times 16$ but the lost velocity variance only by $\times 4$
— in absolute units. But the *fraction* of the total variance lost is
larger for velocity because the velocity variance integral is
dominated by the very large-scale modes that the small box can't
represent.

The same effect shows up mode-by-mode in the **measured spectra**:
$P_\delta(k)$ and $P_v(k)$ at $k > k_{\rm fund}$ are unbiased
estimates of the truth, but the dominant contribution to ψ(r) for
typical $r$ comes from $k \sim 1/r$, so for $r$ comparable to or
larger than $L/2$ the missing low-$k$ modes show up as a depression
of the measured ψ relative to theory. The same depression in ξ(r) is
much smaller because ξ's integrand has $k^2$.

## Demonstration plan

### Step 1 — write `scripts/compute_pv.py`

**Implementation choice.** Three options were considered:

- **(A) Standalone Python.** New `compute_pv.py` mirroring
  `compute_pk.py`; CIC velocity assignment done in NumPy.
- **(B) C extension.** Modify `src/compute_xi.c` to FFT its
  internal CIC velocity grid and emit `pv_*.txt` directly. Requires
  linking FFTW into the C binary.
- **(C) Hybrid.** Add a flag to `compute_xi` to dump the CIC
  velocity grid as a binary file; `compute_pv.py` reads that or
  recomputes from the HDF5.

We pick **(A)**. CIC velocity assignment is ~10 lines of NumPy, the
script runs in seconds for IC-sized boxes (not a hot path), and it
keeps the `pk` and `pv` tools structurally parallel. No C changes.

Mirrors `scripts/compute_pk.py` but for velocity:
- Read positions and velocities from PartType1 in a SWIFT IC HDF5 file.
- CIC-deposit mass and momentum into two parallel grids.
- Form $\mathbf v_{\rm grid}(\mathbf x) = \mathbf p(\mathbf x)/\rho(\mathbf x)$
  (set to 0 for empty cells; rare in IC files).
- For each component $v_i$: FFT, square, deconvolve CIC window
  squared.
- Vector spectrum: $P_v(k) = \sum_i |\hat v_i(k)|^2$ averaged in
  log-spaced $k$ bins.
- Output `pv_<stem>.txt` (k, Pv, Nmodes), units consistent with
  `compute_pk.py`: $k$ in $h/{\rm Mpc}$, $P_v$ in $({\rm km/s})^2 (h/{\rm Mpc})^{-3}$.
- Re-use the interlacing trick (Sefusatti+ 2016) and the CIC window
  deconvolution from `compute_pk.py`.

Naming and CLI mirror `compute_pk.py` so they can be run side-by-side.

### Step 2 — extend `scripts/plot_ic.py`

- Auto-detect `pv_<stem>.txt` in the same directory as `pk_*.txt`.
- Compute theory $P_v(k) = (aHf)^2 P_\delta(k)/k^2$ from CLASS $P(k)$.
- Add a panel to the existing $P(k)$ figure showing measured
  $P_v(k)$ overlaid with theory.

### Step 3 — 4-panel demonstration figure

Generate ICs at three (or more) box sizes (e.g.
$L=25, 250, 1000\,{\rm Mpc}/h$) at fixed $N$ (e.g. 256) and $z$ (e.g. 200).
For each, measure $P_\delta(k)$, $P_v(k)$, $\xi(r)$, and $\psi(r)$.
Build a single figure with four panels (top-left, top-right,
bottom-left, bottom-right):

| | k-space | r-space |
|---|---|---|
| density | $P_\delta(k)$ vs theory, all box sizes overlaid | $\xi(r)$ vs theory |
| velocity | $P_v(k)$ vs theory | $\psi(r)$ vs theory |

Story expected:
- Density panels: small-box runs match theory fine within their k/r
  range (small deviations near box edges).
- Velocity panels: small-box runs visibly sit below theory across a
  wide range of k or r, with the deficit shrinking as $L$ grows.
- The contrast directly illustrates the $1/k^2$ kernel weighting.

Could be either:
- A standalone script under `scripts/` that runs the IC pipeline at
  several box sizes and assembles the figure.
- Or a notebook-style script that reads existing IC outputs and
  plots them.

### Step 4 — update notes

Add a short subsection to `notes/cosmo_ic.tex` (or to the existing
"Starting redshifts and convergence" section) covering:
- The variance integrand difference $k^2 P$ vs $P$.
- The implication for box-size choice.
- A pointer to the new $P_v(k)$ diagnostic and the demonstration figure.

Optionally also update `CLAUDE.md` to document `compute_pv.py` and the
expected output format, alongside `compute_pk.py`.

## Decision points

- **CIC window for velocity**: same $W_{\rm CIC}^2 = \prod_a \mathrm{sinc}^2(k_a\Delta x/2)$
  as for density. The CIC kernel is the same; only the deposited
  quantity differs.
- **Component-summed vs divergence**: simplest is sum-over-components
  $P_v = \sum_i |\hat v_i|^2$ (matches what theory $\propto P_\delta/k^2$
  gives for an irrotational field). Could also report
  $P_\theta = k^2 P_v$ (velocity divergence) but it's redundant.
- **Shot noise**: not subtracting any (CIC velocity grid is a ratio
  $p/\rho$, no direct Poisson contribution). Empty cells contribute
  $v=0$, mildly biasing the spectrum at high $k$; safe at IC redshifts
  where particles are nearly lattice-arranged and no cell is empty.

## Files this will touch / create

- `scripts/compute_pv.py` — new.
- `scripts/plot_ic.py` — extended for `pv_*.txt`.
- `scripts/plot_pk_box_comparison.py` — new (4-panel figure).
- `notes/cosmo_ic.tex` — short subsection on box-size sensitivity.
- `CLAUDE.md` — document `compute_pv.py`.
- `velocity_pk_plan.md` — this file (committable as a planning doc).
