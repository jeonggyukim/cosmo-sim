#!/usr/bin/env python3
"""
compute_pk.py — Estimate the matter power spectrum P(k) from a SWIFT IC HDF5 file.

Method:
  1. Read PartType1 particle positions (in Mpc).
  2. Assign particles to an Ngrid³ mesh using interlaced CIC (two grids offset
     by half a cell, averaged in Fourier space) to suppress aliasing to all orders
     (Sefusatti et al. 2016). Use --no-interlace to disable.
  3. FFT the overdensity field delta(x) = rho(x)/rho_mean - 1.
  4. Correct for the CIC window function W(k).
  5. Average |delta(k)|² in logarithmic k-shells to get P(k).
  6. Optionally extend to smaller scales using the folding technique (Jenkins et al.
     1998): fold particle positions into a sub-box of size L/m, increasing k_Nyquist
     by factor m and reducing Poisson shot noise by m³.  Use --fold M [M ...] to
     specify folding factors; results from all levels are stitched into a single output
     file.  NOTE: folding is only valid at low z (z ≲ 1–2) where nonlinear clustering
     has dispersed particles well beyond their original lattice sites.  At high z the
     near-regular particle lattice aliases its power peak (k = 2π/d) into the folded
     measurement range, producing a spurious upturn above k_Ny.
  7. Output: ASCII table (pk_<stem>.txt).

Use plot_ic.py to visualise the saved table and overlay xi(r), theory, etc.

Usage:
    python compute_pk.py ics_swift_n256_z127_L25.hdf5
    python compute_pk.py ics_swift_n256_z45_L500.hdf5 --ngrid 512 --nkbins 30
    python compute_pk.py ics.hdf5 -o data/pk_custom.txt
    python compute_pk.py ics.hdf5 --fold 4 16        # extend to 16× Nyquist

Output units: k in h/Mpc, P(k) in (Mpc/h)³.

References:
    Hockney & Eastwood (1981) — CIC assignment
    Jing (2005) — alias correction for CIC/TSC
    Sefusatti et al. (2016) — interlaced CIC for alias-free P(k)
    Jenkins et al. (1998) — folding technique for sub-Nyquist P(k)
    Hahn & Abel (2011) — IC validation via P(k)
"""

import argparse
import os

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
        # Use out-of-place evaluation to avoid divide-by-zero at k=0;
        # np.sinc(u) = sin(pi*u)/(pi*u), so sinc(x/pi) = sin(x)/x.
        return np.sinc(x / np.pi)
    return (s(kx) * s(ky) * s(kz)) ** 4


# ---------------------------------------------------------------------------
# Single-level P(k) measurement
# ---------------------------------------------------------------------------

def measure_pk(coords, boxsize_mpc, ngrid, h, interlace, nkbins,
               kmin_mpch=None, kmax_mpch=None):
    """
    Estimate P(k) for particles in a box of size boxsize_mpc.

    Parameters
    ----------
    coords : (N, 3) array
        Particle positions in [0, boxsize_mpc).
    boxsize_mpc : float
        Box side length in Mpc.
    ngrid : int
        FFT mesh size.
    h : float
        Dimensionless Hubble parameter H0/100.
    interlace : bool
        Use interlaced CIC.
    nkbins : int
        Number of logarithmic k-bins.
    kmin_mpch : float or None
        Lower k cut in h/Mpc (default: fundamental mode).
    kmax_mpch : float or None
        Upper k cut in h/Mpc (default: 0.9 × Nyquist).

    Returns
    -------
    dict with keys: k, Pk_raw, Pk_ss, Pk_nodeconv, Pk_err, nmodes, P_shot, knyq_mpch
    """
    N = len(coords)

    kf = 2 * np.pi / boxsize_mpc          # fundamental mode, Mpc⁻¹
    knyq = np.pi * ngrid / boxsize_mpc       # Nyquist, Mpc⁻¹

    # --- CIC assignment ---
    delta = cic_assign(coords, boxsize_mpc, ngrid)

    # --- FFT ---
    delta_k = np.fft.rfftn(delta)   # (ngrid, ngrid, ngrid//2+1)

    # --- k-grid (Mpc⁻¹) ---
    kx = np.fft.fftfreq(ngrid,  d=1.0/ngrid) * kf
    ky = np.fft.fftfreq(ngrid,  d=1.0/ngrid) * kf
    kz = np.fft.rfftfreq(ngrid, d=1.0/ngrid) * kf
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # --- Interlacing ---
    if interlace:
        dx = boxsize_mpc / ngrid
        coords_shifted = (coords + dx / 2) % boxsize_mpc
        delta2 = cic_assign(coords_shifted, boxsize_mpc, ngrid)
        delta2_k = np.fft.rfftn(delta2)
        phase = np.exp(1j * (KX + KY + KZ) * (dx / 2))
        delta_k = 0.5 * (delta_k + phase * delta2_k)

    # --- CIC deconvolution ---
    W2 = cic_window(KX, KY, KZ, knyq)
    W2[0, 0, 0] = 1.0

    Pk_raw = np.abs(delta_k)**2 / W2
    Pk_nodeconv = np.abs(delta_k)**2

    # --- Convert to (Mpc/h)³ ---
    V_box = boxsize_mpc**3
    Pk_modes = Pk_raw * V_box / ngrid**6 * h**3
    Pk_modes_nodeconv = Pk_nodeconv * V_box / ngrid**6 * h**3

    # --- Shot noise ---
    P_shot = V_box / N * h**3   # (Mpc/h)³; = P_shot_original / m³ for folded run

    # --- k in h/Mpc ---
    K_mpch = K / h

    # --- k range ---
    _kmin = kmin_mpch if kmin_mpch is not None else kf / h
    _kmax = kmax_mpch if kmax_mpch is not None else knyq / h * 0.9

    k_edges = np.logspace(np.log10(_kmin), np.log10(_kmax), nkbins + 1)

    k_cen = np.zeros(nkbins)
    Pk_mean = np.zeros(nkbins)
    Pk_err = np.zeros(nkbins)
    Pk_raw_mean = np.zeros(nkbins)
    Pk_nodeconv_mean = np.zeros(nkbins)
    nmodes = np.zeros(nkbins, dtype=int)

    K_flat = K_mpch.ravel()
    Pk_flat = Pk_modes.ravel()
    Pk_flat_nodeconv = Pk_modes_nodeconv.ravel()
    mask = K_flat > 0

    for i, (klo, khi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
        sel = mask & (K_flat >= klo) & (K_flat < khi)
        n = sel.sum()
        if n > 0:
            k_cen[i] = np.mean(K_flat[sel])
            Pk_raw_mean[i] = np.mean(Pk_flat[sel])
            Pk_nodeconv_mean[i] = np.mean(Pk_flat_nodeconv[sel])
            Pk_mean[i] = Pk_raw_mean[i] - P_shot
            Pk_err[i] = np.std(Pk_flat[sel]) / np.sqrt(n)
            nmodes[i] = n

    good = nmodes > 0
    return dict(
        k=k_cen[good],
        Pk_raw=Pk_raw_mean[good],
        Pk_ss=Pk_mean[good],
        Pk_nodeconv=Pk_nodeconv_mean[good],
        Pk_err=Pk_err[good],
        nmodes=nmodes[good],
        P_shot=P_shot,
        knyq_mpch=knyq / h,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Estimate P(k) from a SWIFT IC HDF5 file.")
    parser.add_argument("hdf5", help="SWIFT IC HDF5 file")
    parser.add_argument("--ngrid", type=int, default=None,
                        help="FFT mesh size (default: match particle grid, i.e. N^(1/3))")
    parser.add_argument("--nkbins", type=int, default=60,
                        help="Number of logarithmic k-bins per folding level (default: 30)")
    parser.add_argument("--H0", type=float, default=67.11,
                        help="H0 in km/s/Mpc for unit conversion (default: 67.11)")
    parser.add_argument("--no-interlace", action="store_true", default=False,
                        help="Disable interlaced CIC (default: interlacing on). "
                             "Interlacing uses two grids offset by half a cell to "
                             "suppress aliasing to all orders (Sefusatti et al. 2016).")
    parser.add_argument("--fold", type=int, nargs="+", default=[], metavar="M",
                        help="Folding factors (Jenkins et al. 1998). Each factor M folds particle "
                             "positions into a sub-box of size L/M, extending k_Nyquist by M and "
                             "reducing Poisson shot noise by M³. Results from all levels are "
                             "stitched into one output file. Factors must be given in increasing "
                             "order. WARNING: only meaningful at low z (z ≲ 1-2) where nonlinear "
                             "clustering has dispersed particles well beyond their lattice sites. "
                             "At high z, the near-regular particle lattice aliases its power peak "
                             "k_lattice=2π/d into the folded measurement range.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output txt filename (default: data/pk_<stem>.txt)")
    args = parser.parse_args()

    h = args.H0 / 100.0

    # --- Read HDF5 ---
    with h5py.File(args.hdf5, "r") as f:
        boxsize_mpc = float(np.asarray(f["Header"].attrs["BoxSize"]).reshape(-1)[0])    # Mpc
        redshift = float(np.asarray(f["Header"].attrs["Redshift"]).reshape(-1)[0])
        coords = f["PartType1/Coordinates"][:]          # (N, 3), Mpc

    N = len(coords)
    npart_side = round(N ** (1.0 / 3.0))
    boxsize_mpch = boxsize_mpc * h

    ngrid = args.ngrid if args.ngrid else npart_side
    interlace = not args.no_interlace

    print(f"Particles  : {N} = {npart_side}³")
    print(f"BoxSize    : {boxsize_mpc:.4g} Mpc  =  {boxsize_mpch:.4g} Mpc/h")
    print(f"Redshift   : {redshift:.4g}")
    print(f"FFT grid   : {ngrid}³")
    print(f"k_Nyquist  : {np.pi * ngrid / boxsize_mpc:.4g} Mpc⁻¹  =  "
          f"{np.pi * ngrid / boxsize_mpch:.4g} h/Mpc")
    if args.fold:
        print(f"Fold levels: {[1] + args.fold}  (k_Ny extends to "
              f"{max(args.fold)}× = {max(args.fold) * np.pi * ngrid / boxsize_mpch:.4g} h/Mpc)")
        # Warn if folding is applied at high redshift: IC particles sit nearly on a
        # regular lattice (rms displacement << d = L/N^(1/3)), so folding aliases the
        # lattice power peak at k_lattice = 2π/d into the measurement range, producing
        # a spurious upturn.  Folding is only meaningful once nonlinear clustering has
        # dispersed particles well beyond their lattice sites (z ≲ 1–2).
        d_mpc = boxsize_mpc / npart_side   # mean particle spacing
        k_lattice_mpch = 2 * np.pi / d_mpc / h
        if redshift > 2:
            print(f"\nWARNING: folding at z={redshift:.4g} is unreliable.")
            print(
                f"  Particle spacing d = {d_mpc:.3g} Mpc → k_lattice = {k_lattice_mpch:.3g} h/Mpc")
            print("  At high z, rms displacements << d, so folding aliases the lattice")
            print("  power into the measurement range. Use folding only at z ≲ 1–2.")

    # ---------------------------------------------------------------------------
    # Level m=1: unfolded baseline
    # ---------------------------------------------------------------------------
    print("\n--- Level m=1 (unfolded) ---")
    print(f"Assigning particles to mesh (CIC{'+ interlacing' if interlace else ''})...")
    result_base = measure_pk(coords, boxsize_mpc, ngrid, h, interlace, args.nkbins)
    P_shot = result_base["P_shot"]
    print(f"P_shot     : {P_shot:.4e} (Mpc/h)³")

    all_k = [result_base["k"]]
    all_Pk_raw = [result_base["Pk_raw"]]
    all_Pk_ss = [result_base["Pk_ss"]]
    all_Pk_nd = [result_base["Pk_nodeconv"]]
    all_Pk_err = [result_base["Pk_err"]]
    all_nmodes = [result_base["nmodes"]]
    all_fold_m = [np.ones(len(result_base["k"]), dtype=int)]

    # ---------------------------------------------------------------------------
    # Folded levels
    # ---------------------------------------------------------------------------
    knyq_prev = result_base["knyq_mpch"]

    for m in sorted(args.fold):
        sub_box = boxsize_mpc / m
        kmin_fold = knyq_prev          # start just above previous Nyquist
        kmax_fold = np.pi * ngrid / sub_box / h * 0.9   # 0.9 × this level's Nyquist (h/Mpc)
        P_shot_fold = sub_box**3 / N * h**3

        print(f"\n--- Level m={m} (fold factor {m}, sub-box={sub_box:.4g} Mpc) ---")
        print(f"k range    : [{kmin_fold:.4g}, {kmax_fold:.4g}] h/Mpc")
        print(f"P_shot     : {P_shot_fold:.4e} (Mpc/h)³  (= P_shot / {m}³ = {P_shot:.4e} / {m**3})")
        print("Folding positions...")
        pos_fold = coords % sub_box
        print(f"Assigning particles to mesh (CIC{'+ interlacing' if interlace else ''})...")

        result_fold = measure_pk(pos_fold, sub_box, ngrid, h, interlace, args.nkbins,
                                 kmin_mpch=kmin_fold, kmax_mpch=kmax_fold)

        all_k.append(result_fold["k"])
        all_Pk_raw.append(result_fold["Pk_raw"])
        all_Pk_ss.append(result_fold["Pk_ss"])
        all_Pk_nd.append(result_fold["Pk_nodeconv"])
        all_Pk_err.append(result_fold["Pk_err"])
        all_nmodes.append(result_fold["nmodes"])
        all_fold_m.append(np.full(len(result_fold["k"]), m, dtype=int))

        knyq_prev = result_fold["knyq_mpch"]

    # --- Concatenate all levels ---
    k_out = np.concatenate(all_k)
    Pk_raw_out = np.concatenate(all_Pk_raw)
    Pk_ss_out = np.concatenate(all_Pk_ss)
    Pk_nd_out = np.concatenate(all_Pk_nd)
    Pk_err_out = np.concatenate(all_Pk_err)
    nmodes_out = np.concatenate(all_nmodes)
    fold_m_out = np.concatenate(all_fold_m)

    # --- Save ASCII output ---
    stem = os.path.basename(args.hdf5).replace(".hdf5", "").replace("ics_swift_", "")
    data_dir = os.path.dirname(os.path.abspath(args.hdf5))
    out_txt = args.output or os.path.join(data_dir, f"pk_{stem}.txt")

    fold_str = (f"  fold_factors={[1] + sorted(args.fold)}" if args.fold else "")
    header = (f"P(k) from {args.hdf5}\n"
              f"BoxSize={boxsize_mpch:.4g} Mpc/h  N={N}  ngrid={ngrid}{fold_str}\n"
              f"P_shot(m=1) = {P_shot:.4e} (Mpc/h)^3  (NOT subtracted; shown separately)\n"
              f"Columns: k[h/Mpc]  P_raw[(Mpc/h)^3]  P_shot_sub[(Mpc/h)^3]  "
              f"P_nodeconv[(Mpc/h)^3]  sigma_P  nmodes  fold_m")
    np.savetxt(out_txt,
               np.column_stack([k_out, Pk_raw_out, Pk_ss_out,
                                Pk_nd_out, Pk_err_out, nmodes_out, fold_m_out]),
               header=header,
               fmt=["%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%d", "%d"])
    print(f"\nSaved: {out_txt}")


if __name__ == "__main__":
    main()
