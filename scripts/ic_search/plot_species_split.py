#!/usr/bin/env python3
"""Baryon and cold dark matter spectra under back-scaled and forward initial conditions.

Two ways of setting the initial field differ in what they promise, and the
difference shows up in the acoustic oscillations of the two species.

With back-scaling, the default, CLASS is evaluated at the back-scaling target
redshift (ztarget = 0) and the field is scaled back to the starting redshift with
the growth factor. By z = 0 the baryons have long since fallen into the cold dark
matter potentials, so their transfer functions have nearly converged there;
scaling both by the same factor keeps them nearly equal, and the two species come
out with the same acoustic wiggles.

With ztarget = zstart the transfer functions are taken at the starting redshift
itself and no back-scaling is applied. At z = 200 the baryons have not caught up,
so the two species differ substantially and carry genuinely different acoustic
features. The trade-off is what each promises: the back-scaled field evolves to
the correct z = 0 total matter under an N-body code with no radiation or baryon
physics, while the forward field is the true field at the starting redshift.

Left panel: the measured full-box spectra. Right panel: each measured spectrum
against the theory the run itself wrote, as P/P_theory - 1.

Usage:
    python plot_species_split.py --back DIR --fwd DIR [--out PNG]
"""
import argparse, os
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--back", required=True, help="run directory with ztarget = 0")
ap.add_argument("--fwd", required=True, help="run directory with ztarget = zstart")
ap.add_argument("--lbox", type=float, default=700.0)
ap.add_argument("--out", default=os.path.join(paths.FIGS, "species_split.png"))
A = ap.parse_args()

DSET = {"cdm": "delta_q_cdm", "baryon": "delta_q_baryon"}
THCOL = {"cdm": 2, "baryon": 3}


def measure(rundir):
    """Full-box P(k) per species, and the theory curve the run wrote itself."""
    with h5py.File(f"{rundir}/deltaq.hdf5") as f:
        d = {s: f[DSET[s]][:] for s in DSET}
    N = next(iter(d.values())).shape[0]
    L = A.lbox
    kf = 2*np.pi/L
    ax = np.fft.fftfreq(N, d=1.0/N)*kf
    KX, KY, KZ = np.meshgrid(ax, ax, ax, indexing="ij")
    kk = np.sqrt(KX**2 + KY**2 + KZ**2)
    g = kk > 0
    kny = np.pi*N/L
    nb = 80
    edges = np.linspace(0, kny, nb+1)
    idx = np.digitize(kk[g].ravel(), edges) - 1
    cnt = np.bincount(idx, minlength=nb)[:nb]
    kbin = np.bincount(idx, weights=kk[g].ravel(), minlength=nb)[:nb]/np.maximum(cnt, 1)
    V = L**3

    P = {}
    for s, fld in d.items():
        F = np.abs(np.fft.fftn(fld)/N**3)**2
        P[s] = np.bincount(idx, weights=(V*F)[g].ravel(), minlength=nb)[:nb]/np.maximum(cnt, 1)

    # Evaluate the theory at every mode and bin it exactly as the measurement is
    # binned. Interpolating the theory onto the bin centres instead would smooth
    # the acoustic wiggles that the bin average does not, and the difference shows
    # up as a spurious ripple in the ratio.
    th = np.loadtxt(f"{rundir}/c_input_powerspec.txt")
    Pth = {}
    for s in d:
        P3 = np.zeros_like(kk)
        P3[g] = np.exp(np.interp(np.log(kk[g]), np.log(th[:, 0]),
                                 np.log(th[:, THCOL[s]]*(2*np.pi)**3)))
        Pth[s] = np.bincount(idx, weights=P3[g].ravel(),
                             minlength=nb)[:nb]/np.maximum(cnt, 1)
    ok = (cnt > 0) & (kbin > 0)
    return kbin[ok], {s: P[s][ok] for s in P}, {s: Pth[s][ok] for s in Pth}, kny


kb, Pb, Tb, kny = measure(A.back)
kf_, Pf, Tf, _ = measure(A.fwd)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.4))
STY = {("back", "cdm"): ("C0", "-"), ("back", "baryon"): ("C1", "-"),
       ("fwd", "cdm"): ("C0", "--"), ("fwd", "baryon"): ("C1", "--")}
LBL = {"back": "back-scaled (ztarget = 0)", "fwd": "forward (ztarget = zstart)"}

for tag, k, P in (("back", kb, Pb), ("fwd", kf_, Pf)):
    for s in ("cdm", "baryon"):
        c, ls = STY[(tag, s)]
        a1.loglog(k, P[s], color=c, ls=ls, lw=1.7,
                  label=f"{s}, {LBL[tag]}")
a1.set_xlabel("$k$  [$h$ Mpc$^{-1}$]")
a1.set_ylabel("$P(k)$  [$(\\mathrm{Mpc}/h)^3$]")
a1.set_xlim(kb[0], 0.9*kny)
a1.legend(fontsize=8, framealpha=0.95)
a1.set_title("Measured full-box spectra of the two species", fontsize=10.5)
a1.grid(alpha=0.25)

for tag, k, P, T in (("back", kb, Pb, Tb), ("fwd", kf_, Pf, Tf)):
    for s in ("cdm", "baryon"):
        c, ls = STY[(tag, s)]
        a2.semilogx(k, P[s]/T[s] - 1.0, color=c, ls=ls, lw=1.7,
                    label=f"{s}, {LBL[tag]}")
a2.axhline(0.0, color="0.3", lw=1.0)
a2.set_xlabel("$k$  [$h$ Mpc$^{-1}$]")
a2.set_ylabel("$P(k)\\,/\\,P_{\\rm theory}(k) - 1$")
a2.set_xlim(kb[0], 0.9*kny)
a2.set_ylim(-0.25, 0.25)
a2.legend(fontsize=8, framealpha=0.95, loc="upper left")
a2.set_title("Each against the theory its own run wrote", fontsize=10.5)
a2.grid(alpha=0.25)

fig.suptitle("Back-scaled initial conditions give the two species the same acoustic "
             "features; forward ones do not\n"
             f"$N=256^3$, $L={A.lbox:g}$ Mpc/$h$, $z=200$, same white noise seed, "
             f"full box (no subvolume window)", fontsize=10.8)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(A.out)
print(f"wrote {A.out}\n")

for tag, k, P in (("back-scaled", kb, Pb), ("forward", kf_, Pf)):
    r = np.sqrt(P["baryon"]/P["cdm"])
    m = (k > 0.02) & (k < 0.5)
    print(f"{tag:<12} delta_b/delta_c over k=0.02-0.5: "
          f"{r[m].min():.4f} to {r[m].max():.4f}, "
          f"peak-to-peak {100*(r[m].max()-r[m].min()):.1f}%")
