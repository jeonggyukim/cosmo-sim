#!/usr/bin/env python3
"""
check_ic_dc_mode.py — Verify that the DC (k=0) modes of the displacement
and velocity fields in a MUSIC2-generated IC are zero.

Reports two independent diagnostics:

1. Lagrangian (displacement-based):
     displacement = particle_pos − lattice_pos(ParticleID),
     unwrapped to [−L/2, +L/2] to undo periodic wrapping.
   DC mode = ⟨disp⟩, ⟨vel⟩ over particles.

2. Eulerian (CIC-grid-based):
     ρ(x)     = CIC mass field  → δ(x) = ρ/ρ̄ − 1
     ρv_i(x)  = CIC momentum field → v_i(x) = ρv_i / ρ
   DC mode = ṽ(k=0) = cell-mean of v_grid.  δ̃(k=0) is trivially
   zero by construction (sanity check).

A correctly generated IC has both well below 10⁻⁵ of the RMS.
"""

import argparse
import h5py
import numpy as np


def cic_deposit(pos, L, ngrid, weights=None):
    """CIC deposit into a periodic (ngrid,)*3 grid.  Returns float64 array."""
    grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    dx = L / ngrid
    w = np.ones(len(pos)) if weights is None else weights

    x = pos[:, 0] / dx
    y = pos[:, 1] / dx
    z = pos[:, 2] / dx
    i0 = np.floor(x).astype(np.int64)
    j0 = np.floor(y).astype(np.int64)
    k0 = np.floor(z).astype(np.int64)
    fx, fy, fz = x - i0, y - j0, z - k0
    i0 %= ngrid
    j0 %= ngrid
    k0 %= ngrid
    i1 = (i0 + 1) % ngrid
    j1 = (j0 + 1) % ngrid
    k1 = (k0 + 1) % ngrid

    for (ii, fi) in [(i0, 1 - fx), (i1, fx)]:
        for (jj, fj) in [(j0, 1 - fy), (j1, fy)]:
            for (kk, fk) in [(k0, 1 - fz), (k1, fz)]:
                np.add.at(grid, (ii, jj, kk), w * fi * fj * fk)
    return grid


def check(path, ngrid=None):
    with h5py.File(path, 'r') as f:
        L = float(f['Header'].attrs['BoxSize'])
        pos = f['PartType1/Coordinates'][:]
        vel = f['PartType1/Velocities'][:]
        ids = f['PartType1/ParticleIDs'][:].astype(np.int64) - 1
        mass = f['PartType1/Masses'][:]           # SWIFT internal: 1e10 M_sun
        N = pos.shape[0]
        n = round(N ** (1 / 3))
    m_part = float(mass.mean())                   # 1e10 M_sun per DM particle
    if ngrid is None:
        ngrid = n
    dx_lat = L / n

    print(f'=== {path} ===')
    print(f'  N = {N}, L = {L:.4g} Mpc, lattice dx = {dx_lat:.4g} Mpc')
    print(f'  Mass resolution: m_DM = {m_part:.4g} × 1e10 M_sun '
          f'= {m_part * 1e10:.4g} M_sun/particle')

    # Force-softening recommendation (SWIFT cubic-spline kernel).
    # Literature range: eps_spline ≈ dx/40 (MUSIC/monofonIC, Hahn+2011,
    # Michaux+2020) to dx/25 (GADGET-2/Millennium, Springel 2005; upper
    # bound of Power+2003).  Plummer-equivalent: eps_P ≈ eps_spline/2.8.
    # SWIFT params: Gravity:comoving_DM_softening sits in this range;
    # Gravity:max_physical_DM_softening caps it at ~same value / (1+z_pivot)
    # with default z_pivot = 2.8.
    eps_tight = dx_lat / 40.0
    eps_loose = dx_lat / 25.0
    print('  [Force-softening recommendation (SWIFT cubic-spline)]')
    print(f'    mean spacing dx        = {dx_lat:.4g} Mpc')
    print(f'    eps_spline = dx/40     = {eps_tight:.4g} Mpc  '
          '(MUSIC/monofonIC, tight)')
    print(f'    eps_spline = dx/25     = {eps_loose:.4g} Mpc  '
          '(GADGET-2, loose)')
    print(f'    Plummer-equiv (dx/40)  = {eps_tight / 2.8:.4g} Mpc')
    print(f'    SWIFT Gravity:comoving_DM_softening       ~ {eps_tight:.4g} '
          f'… {eps_loose:.4g} Mpc')
    print(f'    SWIFT Gravity:max_physical_DM_softening   ~ '
          f'{eps_tight / 3.8:.4g} … {eps_loose / 3.8:.4g} Mpc '
          '(z_pivot=2.8)')

    # ---- 1. Lagrangian displacement-based DC check ----------------------
    i = ids // (n * n)
    j = (ids // n) % n
    k = ids % n
    lat = np.stack([(i + 0.5) * dx_lat,
                    (j + 0.5) * dx_lat,
                    (k + 0.5) * dx_lat], axis=1)
    disp = (pos - lat + 0.5 * L) % L - 0.5 * L
    d_mean, d_rms = disp.mean(axis=0), disp.std(axis=0)
    v_mean, v_rms = vel.mean(axis=0), vel.std(axis=0)

    print('  [Lagrangian (displacement-based)]')
    print(f'    <disp>      [Mpc]      = {d_mean}')
    print(f'      / disp_RMS           = {d_mean / d_rms}')
    print(f'    disp RMS    [Mpc]      = {d_rms}')
    print(f'    <vel>  [a km/s]        = {v_mean}')
    print(f'      / vel_RMS            = {v_mean / v_rms}')
    print(f'    vel  RMS [a km/s]      = {v_rms}')

    # ---- 2. Eulerian CIC-grid-based DC check ----------------------------
    rho = cic_deposit(pos, L, ngrid)
    rho_bar = N / ngrid**3
    delta = rho / rho_bar - 1.0
    rhov = np.stack([cic_deposit(pos, L, ngrid, weights=vel[:, a])
                     for a in range(3)], axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        v_grid = np.where(rho[..., None] > 0, rhov / rho[..., None], 0.0)
    delta_dc = delta.mean()
    v_grid_dc = v_grid.reshape(-1, 3).mean(axis=0)

    print(f'  [Eulerian (CIC grid, ngrid={ngrid})]')
    print(f'    δ̃(k=0) / ngrid³       = {delta_dc:.3e}   (≡0 by ρ/ρ̄ − 1)')
    print(f'    δ RMS                  = {delta.std():.3e}')
    v_grid_rms = v_grid.reshape(-1, 3).std(axis=0)
    print(f'    ṽ_grid(k=0) [a km/s]   = {v_grid_dc}')
    print(f'      / v_grid_RMS         = {v_grid_dc / v_grid_rms}')
    print(f'    v_grid RMS [a km/s]    = {v_grid_rms}')
    print()


PREAMBLE = """\
Why check the DC (k=0) mode?
  A nonzero DC mode means the whole simulation volume has a net bulk
  displacement or velocity.  That is unphysical — the displacement field
  is the gradient of a potential whose k=0 mode is undefined and must
  be set to zero by convention.  A leaked DC component would shift every
  correlation-function measurement by a constant, bias bulk-flow
  statistics, and subtly alter growth-factor tests.

Two independent methods, each with its own strengths:

  1. Lagrangian (displacement-based)
     Uses each particle's ParticleID to reconstruct its lattice
     (pre-IC) site, then computes disp = pos − lattice, unwrapped to
     [−L/2, +L/2] to undo SWIFT's periodic wrapping of Coordinates.
     ⟨disp⟩ is the DC mode of the displacement field; ⟨vel⟩ is the
     DC mode of the velocity field.  Exact — each particle contributes
     once with unit weight.  Fails if ParticleID ordering differs from
     the assumed MUSIC2 raster (i = id // n² etc).

  2. Eulerian (CIC-grid-based)
     CIC-deposits particles onto an (ngrid)³ grid to build
       ρ(x)      (mass field)
       ρ v_i(x)  (momentum field)
     then divides to get the cell-averaged velocity v_grid(x).
     ṽ_grid(k=0) = mean over cells.  Robust to particle ordering and
     matches what downstream N-body solvers actually "see", but smooths
     over sub-cell structure (see v_grid RMS vs particle vel RMS).
     δ̃(k=0) is identically zero by construction (ρ/ρ̄ − 1) — printed
     as a sanity check that the CIC deposition conserves mass.

Quantity glossary (printed below):
  <disp>, <vel>         : mean over all particles, per axis
  disp_RMS, vel_RMS     : std over all particles — typical particle-
                          level fluctuation magnitude (the yardstick
                          against which the DC value is judged)
  <disp>/disp_RMS etc.  : dimensionless DC leak; should be ≲ 10⁻⁵
  δ RMS                 : std of CIC density contrast across grid cells
                          — probes the lattice-scale fluctuation level;
                          should grow like D(z) × σ_lattice
  v_grid_RMS            : std of cell-averaged velocity — always ≤
                          vel_RMS; the gap measures sub-cell velocity
                          variance lost to CIC smoothing

Force-softening recommendation:
  Derived from the mean inter-particle spacing dx = L/N using the
  literature range eps = dx/40 (Hahn & Abel 2011; Michaux+ 2020) to
  dx/25 (Springel 2005; upper end of Power+ 2003).  Values are quoted
  for SWIFT's cubic-spline kernel; the Plummer-equivalent scale is
  eps_P ≈ eps_spline / 2.8.  SWIFT switches from comoving to physical
  softening at z_pivot = 2.8 by default, so the physical cap is ~the
  comoving value divided by (1+z_pivot) = 3.8.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('ic_files', nargs='+', help='SWIFT IC HDF5 file(s)')
    ap.add_argument('--ngrid', type=int, default=None,
                    help='CIC grid size per side (default: N^(1/3))')
    ap.add_argument('--explain', action='store_true',
                    help='Print preamble explaining what is measured and why')
    args = ap.parse_args()
    if args.explain:
        print(PREAMBLE)
    for path in args.ic_files:
        check(path, ngrid=args.ngrid)


if __name__ == '__main__':
    main()
