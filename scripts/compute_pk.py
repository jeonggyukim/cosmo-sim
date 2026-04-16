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
  7. Output: ASCII table (pk_<stem>.txt).

Use plot_pk.py to visualise the saved table and overlay xi(r), theory, etc.

Usage:
    python compute_pk.py ics_swift_n256_z127_L25.hdf5
    python compute_pk.py ics_swift_n256_z45_L500.hdf5 --ngrid 512 --nkbins 30
    python compute_pk.py ics.hdf5 -o data/pk_custom.txt

Output units: k in h/Mpc, P(k) in (Mpc/h)³.

References:
    Hockney & Eastwood (1981) — CIC assignment
    Jing (2005) — interlaced CIC deconvolution
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
    parser.add_argument("--interlace", action="store_true", default=False,
                        help="Use interlaced CIC (two grids offset by half a cell) to suppress aliasing")
    parser.add_argument("-o", "--output", default=None,
                        help="Output txt filename (default: data/pk_<stem>.txt)")
    args = parser.parse_args()

    h = args.H0 / 100.0

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

    Pk_raw = np.abs(delta_k)**2 / W2          # deconvolved |delta(k)|²
    Pk_raw_nodeconv = np.abs(delta_k)**2       # no MAS deconvolution (for diagnostics)

    # --- Convert to P(k) in (Mpc/h)³ ---
    V_box_mpc = boxsize_mpc**3
    Pk_modes         = Pk_raw         * V_box_mpc / ngrid**6 * h**3   # (Mpc/h)³
    Pk_modes_nodeconv = Pk_raw_nodeconv * V_box_mpc / ngrid**6 * h**3

    # --- Shot noise in (Mpc/h)³ ---
    P_shot = V_box_mpc / N * h**3   # Poisson shot noise

    # --- Convert k to h/Mpc ---
    K_mpch = K / h    # h/Mpc

    # --- Bin into logarithmic k-shells ---
    kmin_mpch = kf / h
    kmax_mpch = knyq / h * 0.9   # 90% Nyquist

    k_edges = np.logspace(np.log10(kmin_mpch), np.log10(kmax_mpch), args.nkbins + 1)

    k_cen            = np.zeros(args.nkbins)
    Pk_mean          = np.zeros(args.nkbins)
    Pk_err           = np.zeros(args.nkbins)
    Pk_raw_mean      = np.zeros(args.nkbins)
    Pk_nodeconv_mean = np.zeros(args.nkbins)
    nmodes           = np.zeros(args.nkbins, dtype=int)

    K_flat  = K_mpch.ravel()
    Pk_flat = Pk_modes.ravel()
    Pk_flat_nodeconv = Pk_modes_nodeconv.ravel()
    mask    = K_flat > 0

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
    stem = os.path.basename(args.hdf5).replace(".hdf5", "").replace("ics_swift_", "")
    data_dir = os.path.dirname(os.path.abspath(args.hdf5))
    out_txt = args.output or os.path.join(data_dir, f"pk_{stem}.txt")
    header  = (f"P(k) from {args.hdf5}\n"
               f"BoxSize={boxsize_mpch:.4g} Mpc/h  N={N}  ngrid={ngrid}\n"
               f"P_shot = {P_shot:.4e} (Mpc/h)^3  (NOT subtracted; shown separately)\n"
               f"Columns: k[h/Mpc]  P_raw[(Mpc/h)^3]  P_shot_sub[(Mpc/h)^3]  "
               f"P_nodeconv[(Mpc/h)^3]  sigma_P  nmodes")
    np.savetxt(out_txt, np.column_stack([k_cen, Pk_raw_mean, Pk_mean,
                                          Pk_nodeconv_mean, Pk_err, nmodes]),
               header=header, fmt=["%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%d"])
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
