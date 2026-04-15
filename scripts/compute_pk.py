#!/usr/bin/env python3
"""
compute_pk.py — Estimate the matter power spectrum P(k) from a SWIFT IC HDF5 file.

Method:
  1. Read PartType1 particle positions (in Mpc).
  2. Assign particles to an Ngrid³ mesh using Cloud-in-Cell (CIC) interpolation.
  3. FFT the overdensity field delta(x) = rho(x)/rho_mean - 1.
  4. Correct for the CIC window function W(k).
  5. Subtract the Poisson shot noise P_shot = V_box / N_particles.
  6. Average |delta(k)|² in logarithmic k-shells to get P(k).
  7. Output: ASCII table + PNG plot.

Usage:
    python compute_pk.py ics_swift_n256_z127_L25.hdf5
    python compute_pk.py ics_swift_n256_z45_L500.hdf5 --ngrid 512 --nkbins 30
    python compute_pk.py ics.hdf5 --theory class_pk.dat  # overlay CLASS P(k)

Output units: k in h/Mpc, P(k) in (Mpc/h)³.

References:
    Hockney & Eastwood (1981) — CIC assignment
    Jing (2005) — interlaced CIC deconvolution
    Hahn & Abel (2011) — IC validation via P(k)
"""

import argparse
import sys

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# CIC mass assignment
# ---------------------------------------------------------------------------

def cic_assign(pos, boxsize, ngrid):
    """
    Cloud-in-Cell mass assignment.

    Parameters
    ----------
    pos : (N, 3) float array
        Particle positions in [0, boxsize).
    boxsize : float
        Box side length (same units as pos).
    ngrid : int
        Number of grid cells per side.

    Returns
    -------
    delta : (ngrid, ngrid, ngrid) float array
        Overdensity field: rho/rho_mean - 1.
    """
    N = len(pos)
    cellsize = boxsize / ngrid

    # Rescale positions to grid units [0, ngrid)
    g = pos / cellsize  # shape (N, 3)

    # Cell index of the lower-left corner
    i0 = np.floor(g).astype(int)       # integer part
    dx = g - i0                         # fractional offset in [0, 1)

    # CIC weights: each particle contributes to 8 surrounding cells
    rho = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)

    for dxi in range(2):
        wx = (1 - dx[:, 0]) if dxi == 0 else dx[:, 0]
        ix = (i0[:, 0] + dxi) % ngrid
        for dyi in range(2):
            wy = (1 - dx[:, 1]) if dyi == 0 else dx[:, 1]
            iy = (i0[:, 1] + dyi) % ngrid
            for dzi in range(2):
                wz = (1 - dx[:, 2]) if dzi == 0 else dx[:, 2]
                iz = (i0[:, 2] + dzi) % ngrid
                w = wx * wy * wz
                np.add.at(rho, (ix, iy, iz), w)

    # Convert counts to overdensity: delta = rho/rho_mean - 1
    rho_mean = N / ngrid**3
    delta = rho / rho_mean - 1.0
    return delta


# ---------------------------------------------------------------------------
# CIC window function deconvolution
# ---------------------------------------------------------------------------

def cic_window(kx, ky, kz, knyq):
    """
    Squared CIC window function W²(k) = prod_i [sinc(k_i / 2*kNy)]^4.

    Dividing |delta(k)|² by W²(k) corrects for the smoothing introduced
    by CIC assignment.  (sinc here is the unnormalized sinc: sinc(x) = sin(x)/x)
    """
    def s(k):
        x = k / (2 * knyq) * np.pi   # = pi * k_i / (2 * kNy)
        return np.where(np.abs(x) < 1e-10, 1.0, np.sin(x) / x)
    return (s(kx) * s(ky) * s(kz)) ** 4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Estimate P(k) from a SWIFT IC HDF5 file.")
    parser.add_argument("hdf5", help="SWIFT IC HDF5 file")
    parser.add_argument("--ngrid", type=int, default=None,
                        help="FFT mesh size (default: match particle grid, i.e. N^(1/3))")
    parser.add_argument("--nkbins", type=int, default=30,
                        help="Number of logarithmic k-bins (default: 30)")
    parser.add_argument("--H0", type=float, default=67.11,
                        help="H0 in km/s/Mpc for unit conversion (default: 67.11)")
    parser.add_argument("--Omega_m", type=float, default=0.3,
                        help="Matter density (default: 0.3)")
    parser.add_argument("--Omega_b", type=float, default=0.049,
                        help="Baryon density (default: 0.049)")
    parser.add_argument("--theory", default=None,
                        help="Optional CLASS P(k) ASCII file (columns: k[h/Mpc]  P[Mpc/h]^3)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG filename (default: pk_<stem>.png)")
    parser.add_argument("--hankel", action="store_true", default=False,
                        help="Overplot xi(r) from Hankel transform of P(k) (default: off)")
    parser.add_argument("--interlace", action="store_true", default=False,
                        help="Use interlaced CIC (two grids offset by half a cell) to suppress aliasing (default: off)")
    parser.add_argument("--show-nodeconv", action="store_true", default=False,
                        help="Also plot the un-deconvolved P(k) curve (default: off)")
    parser.add_argument("--xi-cic", default=None, metavar="FILE",
                        help="xi_cic output file to overlay on the xi panel (optional)")
    parser.add_argument("--nseeds", type=int, default=8,
                        help="Number of random seeds for subsampling variance estimate (default: 8; set 1 to skip)")
    args = parser.parse_args()

    h = args.H0 / 100.0

    # --- BAO scale: Eisenstein & Hu (1998) fitting formula ---
    Omh2 = args.Omega_m * h**2
    Obh2 = args.Omega_b * h**2
    r_d_Mpc  = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**(3/4))  # Mpc
    r_d_mpch = r_d_Mpc * h                                                    # Mpc/h
    k_BAO    = 2 * np.pi / r_d_mpch                                           # h/Mpc
    print(f"BAO scale  : r_d = {r_d_Mpc:.1f} Mpc = {r_d_mpch:.1f} Mpc/h  "
          f"(k_BAO = {k_BAO:.4f} h/Mpc)  [Eisenstein & Hu 1998]")

    # --- Read HDF5 ---
    with h5py.File(args.hdf5, "r") as f:
        boxsize_mpc = float(f["Header"].attrs["BoxSize"])    # Mpc
        redshift    = float(f["Header"].attrs["Redshift"])
        coords      = f["PartType1/Coordinates"][:]          # (N, 3), Mpc

    N = len(coords)
    npart_side = round(N ** (1.0 / 3.0))
    boxsize_mpch = boxsize_mpc * h   # Mpc/h

    ngrid = args.ngrid if args.ngrid else npart_side

    print(f"Particles  : {N} = {npart_side}³")
    print(f"BoxSize    : {boxsize_mpc:.4g} Mpc  =  {boxsize_mpch:.4g} Mpc/h")
    print(f"FFT grid   : {ngrid}³")
    print(f"k_Nyquist  : {np.pi * ngrid / boxsize_mpc:.4g} Mpc⁻¹  =  "
          f"{np.pi * ngrid / boxsize_mpch:.4g} h/Mpc")

    # --- CIC assignment ---
    print("Assigning particles to mesh (CIC)...")
    delta = cic_assign(coords, boxsize_mpc, ngrid)

    # --- FFT ---
    print("FFT...")
    delta_k = np.fft.rfftn(delta)   # shape (ngrid, ngrid, ngrid//2+1)

    # --- Build k-grid (in Mpc⁻¹) ---
    kf = 2 * np.pi / boxsize_mpc    # fundamental mode in Mpc⁻¹
    knyq = np.pi * ngrid / boxsize_mpc   # Nyquist in Mpc⁻¹

    kx = np.fft.fftfreq(ngrid,  d=1.0/ngrid) * kf   # (ngrid,)
    ky = np.fft.fftfreq(ngrid,  d=1.0/ngrid) * kf
    kz = np.fft.rfftfreq(ngrid, d=1.0/ngrid) * kf   # (ngrid//2+1,)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)   # |k| in Mpc⁻¹

    # --- Interlacing (optional) ---
    # Compute a second CIC grid shifted by half a cell (dx/2 in each dimension),
    # then average the two grids in Fourier space after correcting for the shift phase.
    # This cancels leading alias terms: δ_int(k) = ½[δ₁(k) + e^{ik·Δ} δ₂(k)]
    # where Δ = (dx/2, dx/2, dx/2).  The CIC window deconvolution below still applies.
    if args.interlace:
        print("Interlacing: assigning shifted particles to second mesh...")
        dx = boxsize_mpc / ngrid
        coords_shifted = (coords + dx / 2) % boxsize_mpc
        delta2 = cic_assign(coords_shifted, boxsize_mpc, ngrid)
        delta2_k = np.fft.rfftn(delta2)
        phase = np.exp(1j * (KX + KY + KZ) * (dx / 2))
        delta_k = 0.5 * (delta_k + phase * delta2_k)

    # --- CIC deconvolution ---
    W2 = cic_window(KX, KY, KZ, knyq)
    W2[0, 0, 0] = 1.0   # avoid divide-by-zero at DC

    Pk_raw = np.abs(delta_k)**2 / W2   # deconvolved |delta(k)|²
    Pk_raw_nodeconv = np.abs(delta_k)**2   # no MAS deconvolution

    # --- Convert to P(k) in (Mpc/h)³ ---
    # P(k) = |delta(k)|² * V_box / Ngrid^6  (from DFT convention)
    # Units: [Mpc³] since delta is dimensionless and V_box is in Mpc³
    # Then convert to (Mpc/h)³ by multiplying by h³
    V_box_mpc = boxsize_mpc**3
    Pk_modes = Pk_raw * V_box_mpc / ngrid**6 * h**3   # (Mpc/h)³
    Pk_modes_nodeconv = Pk_raw_nodeconv * V_box_mpc / ngrid**6 * h**3   # no deconv

    # --- Shot noise in (Mpc/h)³ ---
    P_shot = V_box_mpc / N * h**3   # Poisson shot noise

    # --- Convert k to h/Mpc ---
    K_mpch = K / h    # h/Mpc

    # --- Bin into logarithmic k-shells ---
    kmin_mpch = kf / h            # fundamental mode in h/Mpc
    kmax_mpch = knyq / h * 0.9   # go to 90% Nyquist (use --interlace to suppress aliasing near k_Ny)

    k_edges = np.logspace(np.log10(kmin_mpch), np.log10(kmax_mpch), args.nkbins + 1)

    k_cen    = np.zeros(args.nkbins)
    Pk_mean  = np.zeros(args.nkbins)
    Pk_err   = np.zeros(args.nkbins)
    nmodes   = np.zeros(args.nkbins, dtype=int)

    # Flatten arrays for binning (exclude DC mode k=0)
    K_flat  = K_mpch.ravel()
    Pk_flat = Pk_modes.ravel()
    Pk_flat_nodeconv = Pk_modes_nodeconv.ravel()
    mask    = K_flat > 0

    Pk_raw_mean = np.zeros(args.nkbins)   # deconvolved, no shot subtraction
    Pk_nodeconv_mean = np.zeros(args.nkbins)   # no deconvolution, no shot subtraction

    for i, (klo, khi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
        sel = mask & (K_flat >= klo) & (K_flat < khi)
        n = sel.sum()
        if n > 0:
            k_cen[i]            = np.mean(K_flat[sel])
            Pk_raw_mean[i]      = np.mean(Pk_flat[sel])
            Pk_nodeconv_mean[i] = np.mean(Pk_flat_nodeconv[sel])
            Pk_mean[i]          = Pk_raw_mean[i] - P_shot
            Pk_err[i]           = np.std(Pk_flat[sel]) / np.sqrt(n)
            nmodes[i]           = n

    # Keep only populated bins
    good = nmodes > 0
    k_cen, Pk_mean, Pk_raw_mean, Pk_nodeconv_mean, Pk_err, nmodes = (
        k_cen[good], Pk_mean[good], Pk_raw_mean[good], Pk_nodeconv_mean[good],
        Pk_err[good], nmodes[good])

    # --- Save ASCII output ---
    import os
    stem = os.path.basename(args.hdf5).replace(".hdf5", "").replace("ics_swift_", "")
    data_dir = os.path.dirname(os.path.abspath(args.hdf5))
    out_txt = os.path.join(data_dir, f"pk_{stem}.txt")
    header  = (f"P(k) from {args.hdf5}\n"
               f"BoxSize={boxsize_mpch:.4g} Mpc/h  N={N}  ngrid={ngrid}\n"
               f"P_shot = {P_shot:.4e} (Mpc/h)^3  (NOT subtracted; shown separately)\n"
               f"Columns: k[h/Mpc]  P_raw[(Mpc/h)^3]  P_shot_sub[(Mpc/h)^3]  sigma_P  nmodes")
    np.savetxt(out_txt, np.column_stack([k_cen, Pk_raw_mean, Pk_mean, Pk_err, nmodes]),
               header=header, fmt=["%.6e", "%.6e", "%.6e", "%.6e", "%d"])
    print(f"Saved: {out_txt}")

    # --- Theory xi(r) from CLASS P(k) via Hankel transform ---
    xi_theory_r = xi_theory_xi = None
    if args.theory:
        from mcfit import P2xi
        kt, Pt = np.loadtxt(args.theory, comments='#', unpack=True)
        xi_theory_r, xi_theory_xi = P2xi(kt, l=0)(Pt, extrap=True)

    # --- Measured xi(r) via Hankel transform of P(k) (optional) ---
    xi_meas = None
    if args.hankel:
        trapz = np.trapezoid
        r_xi = np.logspace(-1, np.log10(boxsize_mpch / 2), 300)
        xi_meas = np.zeros(len(r_xi))
        for i, r in enumerate(r_xi):
            x = k_cen * r
            sinc = np.where(np.abs(x) < 1e-8, 1.0, np.sin(x) / x)
            xi_meas[i] = trapz(Pk_raw_mean * sinc * k_cen**2, k_cen) / (2 * np.pi**2)

    # --- Plot ---
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax, ax2 = axes

    # ---- Left panel: P(k) ----
    if getattr(args, 'show_nodeconv', False):
        ax.loglog(k_cen, Pk_nodeconv_mean, 's--', ms=3, lw=1.0, color='C3', alpha=0.7,
                  label='no MAS deconv')
    ax.loglog(k_cen, Pk_raw_mean, 'o-', ms=4, lw=1.2, color='C0',
              label='MAS deconvolved')
    pos_shot_sub = Pk_mean > 0
    ax.loglog(k_cen[pos_shot_sub], Pk_mean[pos_shot_sub], '^-', ms=4, lw=1.2, color='C2',
              label='MAS deconvolved + shot subtracted')

    if args.theory:
        ax.loglog(kt, Pt, 'k-', lw=1.2, label='theory (CLASS)')

    knyq_mpch  = knyq / h
    kfund_mpch = kf / h

    ax.axvline(kfund_mpch, color='C2',   ls=':', lw=1.0,
               label=fr'$k_f = 2\pi/L$ = {kfund_mpch:.2g} $h$/Mpc')
    ax.axvline(knyq_mpch,  color='gray', ls='--', lw=1.0,
               label=fr'$k_\mathrm{{Ny}}$ = {knyq_mpch:.2g} $h$/Mpc')
    ax.axhline(P_shot, color='orange', ls='--', lw=1.0,
               label=fr'$P_\mathrm{{shot}}$ = {P_shot:.2g} (Mpc/$h$)$^3$')

    ax.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
    ax.set_ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]')
    ax.set_title(f'{npart_side}³, L={boxsize_mpch:.4g} Mpc/h, z={redshift:.4g}')
    ax.legend(fontsize=8)

    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(2 * np.pi / np.array(ax.get_xlim()))
    ax_top.invert_xaxis()
    ax_top.set_xlabel(r'$\lambda = 2\pi/k$ [Mpc/$h$]')

    # ---- Right panel: xi(r) ----
    if xi_theory_r is not None:
        mask_t = (xi_theory_r > 0) & (xi_theory_xi > 0) & (xi_theory_r < boxsize_mpch / 2)
        ax2.loglog(xi_theory_r[mask_t], xi_theory_xi[mask_t],
                   'k-', lw=1.2, label='theory (CLASS)')

    if xi_meas is not None:
        ax2.loglog(r_xi[xi_meas > 0], xi_meas[xi_meas > 0], 'o-', ms=4, lw=1.2,
                   color='C0', alpha=0.5, label='measured (Hankel transform of $P(k)$)')

    # Overplot Corrfunc xi(r) if the output file exists
    # Columns: r_avg r_low r_high xi npairs  (r_avg=0 unless need_avg_sep=1)
    xi_corrfunc_file = os.path.join(data_dir, f"xi_{stem}.txt")
    if os.path.exists(xi_corrfunc_file):
        cf = np.loadtxt(xi_corrfunc_file, comments='#')
        r_low, r_high, xi_cf, npairs = cf[:, 1], cf[:, 2], cf[:, 3], cf[:, 4]
        r_mid = np.sqrt(r_low * r_high) * h   # geometric mean, Mpc → Mpc/h
        xi_poisson_err = (1 + xi_cf) / np.sqrt(np.maximum(npairs, 1))

        # Subsampling variance: re-run compute_xi with different random seeds
        # and measure the scatter across runs.
        xi_subsample_std = np.zeros_like(xi_cf)
        nseeds = getattr(args, 'nseeds', 8)
        rbins_file = os.path.join(data_dir, f"rbins_{stem}.txt")
        repo_root = os.path.dirname(data_dir)
        xi_binary = os.path.join(repo_root, 'compute_xi')
        if nseeds > 1 and os.path.exists(xi_binary) and os.path.exists(rbins_file):
            import subprocess
            xi_runs = []
            for seed in range(1, nseeds + 1):
                result = subprocess.run(
                    [xi_binary, args.hdf5, rbins_file, '4', '-s', str(seed)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    lines = [l for l in result.stdout.splitlines() if not l.startswith('#')]
                    vals = np.array([list(map(float, l.split())) for l in lines if l.strip()])
                    if vals.shape[0] == len(xi_cf):
                        xi_runs.append(vals[:, 3])
            if len(xi_runs) >= 2:
                xi_subsample_std = np.std(xi_runs, axis=0, ddof=1)

        xi_err = np.sqrt(xi_poisson_err**2 + xi_subsample_std**2)
        pos = xi_cf > 0
        # Only show Corrfunc if at least half the bins have positive ξ;
        # otherwise the measurement is noise-dominated and misleading.
        if pos.sum() >= len(xi_cf) / 2:
            label = 'measured (Corrfunc)'
            if xi_subsample_std.any():
                label += f' ±{nseeds}-seed'
            ax2.errorbar(r_mid[pos], xi_cf[pos],
                         yerr=xi_err[pos],
                         fmt='s-', ms=4, lw=1.2, elinewidth=0.8, capsize=2,
                         color='C1', label=label)

    # xi_cic overlay (grid autocorrelation estimator)
    xi_cic_file = getattr(args, 'xi_cic', None)
    if xi_cic_file is None:
        # auto-detect: data/xi_cic_<stem>.txt alongside the IC file
        xi_cic_auto = os.path.join(data_dir, f"xi_cic_{stem}.txt")
        if os.path.exists(xi_cic_auto):
            xi_cic_file = xi_cic_auto
    if xi_cic_file is not None and os.path.exists(xi_cic_file):
        cic = np.loadtxt(xi_cic_file, comments='#')
        # columns: r_avg r_low r_high xi DD DR RR
        r_cic_low  = cic[:, 1]
        r_cic_high = cic[:, 2]
        xi_cic     = cic[:, 3]
        r_cic_mid  = np.sqrt(r_cic_low * r_cic_high) * h   # Mpc → Mpc/h
        pos_c = xi_cic > 0
        neg_c = xi_cic < 0
        ax2.loglog(r_cic_mid[pos_c], xi_cic[pos_c], '^-', ms=4, lw=1.2,
                   color='C4', label='measured (CIC grid, $\\xi>0$)')
        if neg_c.any():
            ax2.loglog(r_cic_mid[neg_c], np.abs(xi_cic[neg_c]), '^--', ms=4, lw=0.8,
                       color='C4', alpha=0.4, label=r'measured (CIC grid, $\xi<0$)')

    # Mark L/3 — maximum reliable separation for periodic box
    ax2.axvline(boxsize_mpch / 3, color='gray', ls='--', lw=1.0,
                label=fr'$L/3$ = {boxsize_mpch/3:.4g} Mpc/$h$')
    # Mark mean particle spacing
    dx_mpch = boxsize_mpch / npart_side
    ax2.axvline(dx_mpch, color='C2', ls=':', lw=1.0,
                label=fr'$\Delta x$ = {dx_mpch:.2g} Mpc/$h$')

    ax2.set_xlabel(r'$r$ [Mpc/$h$]')
    ax2.set_ylabel(r'$\xi(r)$')
    ax2.set_title(r'Correlation function $\xi(r)$')
    ax2.legend(fontsize=8)

    fig.tight_layout()

    out_png = args.output or f"pk_{stem}.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
