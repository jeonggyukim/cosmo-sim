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
ap.add_argument("--back", default=None,
                help="run directory with ztarget = 0; omit to show the forward run alone")
ap.add_argument("--fwd", required=True, help="run directory with ztarget = zstart")
ap.add_argument("--lbox", type=float, default=700.0)
ap.add_argument("--out", default=os.path.join(paths.FIGS, "species_split.png"))
A = ap.parse_args()

DSET = {"cdm": "delta_q_cdm", "baryon": "delta_q_baryon"}
THCOL = {"cdm": 2, "baryon": 3}


def read_ztarget(rundir):
    """The ztarget the run actually used, taken from its own config."""
    import re
    for line in open(f"{rundir}/c.conf"):
        m = re.match(r"\s*ztarget\s*=\s*([0-9.eE+-]+)", line)
        if m:
            return float(m.group(1))
    return 0.0


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
    # Log-spaced bins: the plot is logarithmic in k, and linear bins would put
    # only one or two of them below k = 0.03 while crowding the small scales.
    nb = 26
    kfund = 2*np.pi/L
    edges = np.logspace(np.log10(0.9*kfund), np.log10(kny), nb+1)
    idx = np.digitize(kk[g].ravel(), edges) - 1
    inb = (idx >= 0) & (idx < nb)
    idx = idx[inb]
    cnt = np.bincount(idx, minlength=nb)[:nb]
    kbin = np.bincount(idx, weights=kk[g].ravel()[inb], minlength=nb)[:nb]/np.maximum(cnt, 1)
    V = L**3

    P = {}
    for s, fld in d.items():
        F = np.abs(np.fft.fftn(fld)/N**3)**2
        P[s] = np.bincount(idx, weights=(V*F)[g].ravel()[inb], minlength=nb)[:nb]/np.maximum(cnt, 1)

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
        Pth[s] = np.bincount(idx, weights=P3[g].ravel()[inb],
                             minlength=nb)[:nb]/np.maximum(cnt, 1)
    # A bin needs enough modes for its average to mean anything. At the fundamental
    # there are only six modes in the whole box, so the first bins are dropped
    # rather than plotted as noise.
    ok = (cnt >= 12) & (kbin > 0)
    # The raw table too, for drawing the theory at its own resolution rather than
    # at the resolution of the measurement's bins.
    fine = {s: (th[:, 0], th[:, THCOL[s]]*(2*np.pi)**3) for s in d}
    zs = 0.0
    for line in open(f"{rundir}/c.conf"):
        if line.strip().startswith("zstart"):
            zs = float(line.split("=")[1].split("#")[0])
    return (kbin[ok], {s: P[s][ok] for s in P}, {s: Pth[s][ok] for s in Pth},
            kny, fine, N, zs)


runs = [("fwd", A.fwd)] + ([("back", A.back)] if A.back else [])
M = {tag: measure(d) for tag, d in runs}
kny = next(iter(M.values()))[3]

# The measurable range is the fundamental mode to the Nyquist wavenumber, with a
# little margin at each end so neither limit sits on the frame.
KFUND = 2*np.pi/A.lbox
KNY = kny
KLO, KHI = 0.7*KFUND, 1.4*KNY

plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5,
                     "axes.labelsize": 12.5, "legend.fontsize": 10.5,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 6.0))
COL = {"cdm": "C0", "baryon": "C1"}
LS = {"fwd": "-", "back": "--"}
ZT = {tag: read_ztarget(d) for tag, d in runs}
LBL = {tag: (f"forward, ztarget = {ZT[tag]:g}" if ZT[tag] > 0
             else f"back-scaled, ztarget = {ZT[tag]:g}") for tag, _ in runs}

for tag, _ in runs:
    k, P, T, _, fine = M[tag][:5]
    for s in ("cdm", "baryon"):
        # Theory drawn from its own table, which samples k far more finely than
        # the measurement's bins and so shows the acoustic wiggles fully resolved.
        kt, Pt = fine[s]
        mt = (kt >= KLO) & (kt <= KHI)
        a1.loglog(kt[mt], Pt[mt], color=COL[s], lw=4.0, alpha=0.25,
                  solid_capstyle="round",
                  label="linear theory" if (s == "cdm" and tag == runs[0][0]) else None)
        # One marker per band, so it is clear that the measurement exists only at
        # these wavenumbers and the line between them is drawn, not measured.
        a1.loglog(k, P[s], color=COL[s], ls=LS[tag], lw=1.4, marker="o", ms=3.6,
                  mfc="white", mew=1.2, label=f"{s}, {LBL[tag]}")
        a2.semilogx(k, P[s]/T[s] - 1.0, color=COL[s], ls=LS[tag], lw=1.4,
                    marker="o", ms=3.6, mfc="white", mew=1.2,
                    label=f"{s}, {LBL[tag]}")

for ax in (a1, a2):
    for kv, lab in ((KFUND, "$k_{\\rm fund}$"), (KNY, "$k_{\\rm Ny}$")):
        ax.axvline(kv, color="0.55", lw=1.0, ls=":")
    ax.text(KFUND, 0.02, " $k_{\\rm fund}$", transform=ax.get_xaxis_transform(),
            fontsize=10, color="0.4", ha="left")
    ax.text(KNY, 0.02, " $k_{\\rm Ny}$", transform=ax.get_xaxis_transform(),
            fontsize=10, color="0.4", ha="left")

a1.set_xlabel("$k$  [$h$ Mpc$^{-1}$]")
a1.set_ylabel("$P(k)$  [$(\\mathrm{Mpc}/h)^3$]")
a1.set_xlim(KLO, KHI)
a1.legend(framealpha=0.95)
a1.set_title("Measured full-box spectra, with linear theory beneath")
a1.grid(alpha=0.25)

a2.axhline(0.0, color="0.3", lw=1.0)
a2.set_xlabel("$k$  [$h$ Mpc$^{-1}$]")
a2.set_ylabel("$P(k)\\,/\\,P_{\\rm theory}(k) - 1$")
a2.set_xlim(KLO, KHI)
a2.set_ylim(-0.25, 0.25)
a2.legend(framealpha=0.95, loc="upper left")
a2.set_title("Each against the theory its own run wrote")
a2.grid(alpha=0.25)

N3 = int(round(next(iter(M.values()))[5]))
zs = next(iter(M.values()))[6]
fig.suptitle("Taking the transfer functions at the starting redshift\n"
             "separates the two species\n"
             f"$N={N3}^3$, $L={A.lbox:g}$ Mpc/$h$, $z_{{\\rm start}}={zs:g}$, "
             f"monofonIC "
             + ", ".join(f"[cosmology] ztarget = {ZT[t]:g}" for t, _ in runs)
             + ", full box", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(A.out)
print(f"wrote {A.out}\n")

for tag, _ in runs:
    k, P, T = M[tag][0], M[tag][1], M[tag][2]
    r = np.sqrt(P["baryon"]/P["cdm"])
    m = (k > 0.02) & (k < 0.5)
    print(f"{LBL[tag]:<32} delta_b/delta_c over k=0.02-0.5: "
          f"{r[m].min():.4f} to {r[m].max():.4f}, "
          f"peak-to-peak {100*(r[m].max()-r[m].min()):.1f}%")
