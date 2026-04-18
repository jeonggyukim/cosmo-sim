# Baryon + CDM IC Generation — Plan

Working roadmap for extending `cosmo-pipeline` from pure-CDM ICs to
joint baryon + CDM ICs.  Three layers; can be pursued independently.

## 1. Theory write-up — fill `notes/cosmo_ic.tex` §9 (`sec:baryons_cdm`)

Currently a placeholder.  To cover:

- **Two-fluid linear system (c–b).**  Derive $\delta_c$ and $\delta_b$
  growth ODEs from the full continuity/Euler/Poisson system with
  $-\nabla p / \rho_b$ and adiabatic closure
  $c_s^2 = \gamma k_B T_b / (\mu m_H)$.  Express as the growing
  "total matter" mode $\delta_m = f_c \delta_c + f_b \delta_b$ and the
  decaying "compensated" mode $\delta_{cb} = \delta_c - \delta_b$
  (Somogyi & Smith 2010; Shoji & Komatsu 2009).
- **Jeans scale at high $z$.**  Analytic $\lambda_J^{\rm com}(z)$ using
  the post-recombination $T_b(z)$ history (Compton-coupled to $T_\gamma$
  until $z \sim 150$, then adiabatic $T_b \propto (1+z)^2$).  Typical
  numbers for starting redshifts $z \sim 100$–200: $k_J \sim 50$–100
  h/Mpc — above Nyquist for current box setups, so Jeans suppression
  is invisible on the grid but still encoded in $T_b(k, z_{\rm start})$
  from CLASS.
- **Relative streaming velocity $v_{bc}$** (Tseliakhovich & Hirata 2010).
  Supersonic $\sim 30$ km/s at recombination, relevant down to
  $z \sim 40$; modulates small-scale power and first-star formation.
- **Compensated isocurvature perturbation (CIP).**  What it is, when
  to enable it in IC generation, and when the pure adiabatic mode is
  sufficient.

## 2. Pipeline changes

- **`scripts/make_music_conf.py`**: add `--baryons` flag that sets
  `baryons = yes` and `baryons_density = separate` (MUSIC2) or the
  `monofonIC` equivalent.  Ensure the resulting CLASS `.ini` requests
  both $T_c$ and $T_b$ in the output.
- **`run_pipeline.sh`**: handle PartType0 (gas) alongside PartType1
  (DM); set SWIFT IC metadata (ParticleIDs, InternalEnergy or entropy
  field, correct masses per species using
  $\Omega_c$ vs $\Omega_b$).
- **`compute_pk.py` / `plot_ic.py`**: support per-species spectra
  $P_{cc}$, $P_{bb}$ and the cross spectrum $P_{cb}(k)$.  Overlay
  CLASS predictions for all three.

## 3. Validation

- **`scripts/check_ic_dc_mode.py`**: extend to both species.
- **New diagnostic** (most important): recover $T_b(k)/T_c(k)$ from
  the IC and compare to CLASS.  This is the real test that the
  two-species sampling is *coherent* — the same random seed must
  produce correlated $\delta_c$ and $\delta_b$ realisations, not
  independent draws.  Decorrelation here silently breaks the
  compensated mode and the relative velocity statistics.

## Main tradeoff

Doing (2) seriously means committing to SWIFT hydro plumbing
(smoothing lengths, neighbour counts, entropy/energy choice) and
likely switching to `monofonIC` for the baryon path — its two-species
support is more recent and better tested than MUSIC2's.

If the near-term goal is only to *study* baryon ICs (not run
production hydro), start with (1) + a minimal (3): generate
two-species ICs, validate the transfer functions on the output, and
defer the SWIFT hydro integration until we actually want to evolve
them.

## Suggested first concrete task

Either:

- Flesh out §9 in `cosmo_ic.tex` (theory; no code changes), or
- Prototype `make_music_conf.py --baryons` and inspect the resulting
  MUSIC2 config + HDF5 output by eye — gives a fast read on whether
  MUSIC2 or monofonIC is the right baseline before investing in the
  validation tooling.
