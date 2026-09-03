#!/usr/bin/env python3
"""Whole box against pencil zoom-in region, for each species, in P(k) and in xi.

Two figures with the same encoding, so they can be read as a pair: colour is the
species, line style is the region, and the pale thick curve beneath each pair is
the linear theory that species is measured against.

The pair is the argument. Masking is a multiplication in configuration space, so
what a subvolume measures is M(x) delta(x) with M its indicator function. The
correlation function is a ratio of pair counts,

    xi(r) = sum_x M(x) M(x+r) delta(x) delta(x+r) / sum_x M(x) M(x+r),

which is grid Landy-Szalay in the infinite-random limit, written as a convolution
after Slepian & Eisenstein (2016). The mask contributes to the numerator and the
denominator alike and cancels, so the pencil and the box are unbiased for the
same xi. The power spectrum is the transform of that same product, and a product
transforms to a convolution, which mixes modes and cannot be undone per mode. So
the P(k) figure carries two theory curves per species, the raw one and the one
convolved with the pencil window, while the xi figure carries only one.

Neither measurement is wrong. Both estimators are unbiased; they are unbiased
for different things. The pencil P(k) estimates the window-convolved spectrum,
and only looks low if it is held against the raw theory instead.

That is why selecting a seed on a pencil's P(k) conditions the field. Matching
the unconvolved theory asks a realization to cancel a deficit the geometry put
there, and only an upward fluctuation of the region's large-scale power can do
it. The same criterion on xi asks for nothing of the kind.

Usage:
    python plot_box_vs_pencil.py --data DIR [--out-pk PNG] [--out-xi PNG]
"""
import argparse, glob, os
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "xi128"))
ap.add_argument("--species", nargs="+", default=None,
                help="species to show, in order; default is every species present")
ap.add_argument("--compare", default="matter",
                help="species used for the whole-box against pencil row")
ap.add_argument("--rmax", type=float, default=None,
                help="largest separation shown, in Mpc/h. Default is the image of "
                     "the P(k) panel's low-k edge under r = 2 pi / k, clipped to the "
                     "largest separation the sweep measured")
ap.add_argument("--out-pk", default=os.path.join(paths.FIGS, "pk_box_vs_pencil.png"))
ap.add_argument("--out-xi", default=os.path.join(paths.FIGS, "xi_box_vs_pencil.png"))
A = ap.parse_args()

files = sorted(glob.glob(f"{A.data}/chunk_*.hdf5"))
if not files:
    raise SystemExit(f"no chunk_*.hdf5 under {A.data}")

with h5py.File(files[0]) as f:
    if "xi_full" not in f:
        raise SystemExit(f"{files[0]} carries no xi: rerun the sweep with --xi")
    present = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
    r, k = f["r"][:], f["k"][:]
    N, L = int(f.attrs["N"]), float(f.attrs["L"])
    frac, kny = int(f.attrs["frac"]), float(f.attrs["kny"])
    P_TH, P_WIN = f["P_theory"][:], f["P_win"][:]
    nmodes = (f["nmodes"][:].astype(float) if "nmodes" in f else None)
    _df = f.attrs.get("dofixing", "no")
    FIXED = str(_df.decode() if isinstance(_df, bytes) else _df).lower() in ("yes", "true", "1")

SP = [s for s in (A.species or present) if s in present]
if not SP:
    raise SystemExit(f"none of {A.species} present; the file has {present}")
IDX = {s: present.index(s) for s in SP}

XF, XP, PF, PP = [], [], [], []
for fn in files:
    with h5py.File(fn) as f:
        XF.append(f["xi_full"][:]);  XP.append(f["xi_pencil"][:])
        PF.append(f["P_full"][:]);   PP.append(f["P_pencil"][:])
XF = np.concatenate(XF); XP = np.concatenate(XP)
PF = np.concatenate(PF); PP = np.concatenate(PP)
nseed, npen = XP.shape[0], XP.shape[2]
width = L/frac
KFUND = 2*np.pi/L
KLO, KHI = 0.7*KFUND, 1.4*kny
# The separation axis is the image of the wavenumber axis under r = 2 pi / k, so
# the two rows of a figure cover the same physical scales with the same margins:
# the left edge is the Nyquist wavenumber and the right edge the fundamental,
# each carrying the same factor of margin as the P(k) panels. The association is
# a convention rather than an identity, since xi and P are integral transforms of
# one another and no single k maps to a single r, but it is what lets the two be
# read against each other. The upper end is then clipped to the largest
# separation measured, which is well inside it.
RLO = 2*np.pi/KHI
RHI = min(2*np.pi/KLO, r.max()) if A.rmax is None else A.rmax

if nmodes is None:
    # Chunks written before nmodes was recorded: rebuild it from the sweep's own
    # binning rule, which depends on nothing but N and L.
    _ka = np.fft.fftfreq(N, 1.0/N)*KFUND
    _kk = np.sqrt(sum(g**2 for g in np.meshgrid(_ka, _ka, _ka, indexing="ij")))
    _ib = np.digitize(_kk.ravel(), np.arange(0.5, N/2 + 1.0)*KFUND)
    nmodes = np.array([(_ib == i + 1).sum() for i in range(len(k))], float)


def xi_theory(Pth):
    """Linear xi on the measurement's own bins, from the same theory spectrum.

    With P(k) = V <|delta_k|^2> and delta_k the transform divided by N^3,
    xi(r) = (1/V) sum_k P(k) exp(i k.r) = (N^3/V) ifftn(P)(r), which reduces at
    r = 0 to the variance, as the measured curve does. Binning it the way the
    measurement is binned keeps the two comparable bin by bin.
    """
    ka = np.fft.fftfreq(N, d=1.0/N)*KFUND
    k3 = np.sqrt(sum(g**2 for g in np.meshgrid(ka, ka, ka, indexing="ij")))
    P3 = np.zeros_like(k3)
    pos = k3 > 0
    P3[pos] = np.exp(np.interp(np.log(k3[pos]), np.log(k), np.log(Pth)))
    xi3 = np.real(np.fft.ifftn(P3))*N**3/L**3
    ra = np.minimum(np.arange(N), N - np.arange(N))*(L/N)
    r3 = np.sqrt(sum(g**2 for g in np.meshgrid(ra, ra, ra, indexing="ij")))
    edge = np.linspace(0.0, r.max() + 0.5*(r[1] - r[0]), len(r) + 1)
    idx = np.digitize(r3.ravel(), edge) - 1
    inb = (idx >= 0) & (idx < len(r))
    cnt = np.bincount(idx[inb], minlength=len(r))[:len(r)]
    return (np.bincount(idx[inb], weights=xi3.ravel()[inb], minlength=len(r))[:len(r)]
            / np.maximum(cnt, 1))


plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5,
                     "axes.labelsize": 12.5, "legend.fontsize": 10.5,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})
COL = {"cdm": "C0", "baryon": "C1", "matter": "0.25"}
NAME = {"cdm": "CDM", "baryon": "baryon", "matter": "total matter"}

# One species carries the box-against-pencil comparison. Showing all three there
# put six curves in a panel to make a point that does not depend on species at
# all: the mask acts on geometry, so it treats every species alike.
ONE = A.compare if A.compare in IDX else SP[0]
BOXC, PENC = "C0", "C3"

SUB = (f"$N={N}^3$, $L={L:g}$ Mpc/$h$, pencil {width:g} Mpc/$h$ across, "
       f"{nseed} realization{'s' if nseed != 1 else ''} $\\times$ {npen} "
       f"subvolume{'s' if npen != 1 else ''}"
       + (", amplitudes fixed to linear theory (DoFixing = yes)" if FIXED else ""))
MK = dict(lw=1.3, ms=2.8, mfc="white", mew=0.9)


def decorate_k(a):
    a.set_xlabel("$k$  [$h$ Mpc$^{-1}$]")
    a.set_xlim(KLO, KHI)
    for kv, lab in ((KFUND, "$k_{\\rm fund}$"), (kny, "$k_{\\rm Ny}$")):
        a.axvline(kv, color="0.55", lw=1.0, ls=":")
        a.text(kv, 0.02, " " + lab, transform=a.get_xaxis_transform(),
               fontsize=10, color="0.4", ha="left")
    a.grid(alpha=0.25)


def decorate_r(a):
    a.set_xlabel("$r$  [Mpc/$h$]")
    a.set_xlim(RLO, RHI)
    if RLO < width < RHI:
        a.axvline(width, color="0.55", lw=1.0, ls=":")
        a.text(width, 0.02, " pencil width", transform=a.get_xaxis_transform(),
               fontsize=10, color="0.4", ha="left")
    a.grid(alpha=0.25)


sig = np.sqrt(2.0/np.maximum(nmodes, 1)/nseed)
BANDLAB = (r"scatter without fixing, $\sqrt{2/N_{\rm modes}}$" if FIXED
           else r"expected scatter, $\sqrt{2/N_{\rm modes}n_{\rm real}}$")

# ---------------------------------------------------------------- P(k) -------
fig, ax = plt.subplots(2, 2, figsize=(13.6, 10.4))

a = ax[0, 0]
for sp in SP:
    i = IDX[sp]
    a.loglog(k, P_TH[i], color=COL[sp], lw=4.0, alpha=0.22, solid_capstyle="round")
    a.loglog(k, PF[:, i].mean(0), color=COL[sp], marker="o", label=NAME[sp], **MK)
a.set_ylabel("$P(k)$  [$(\\mathrm{Mpc}/h)^3$]")
a.set_title("Whole box, each species, with its own theory beneath")
a.legend(framealpha=0.95)
decorate_k(a)

a = ax[0, 1]
a.fill_between(k, -sig, sig, color="0.75", alpha=0.45, lw=0, zorder=0, label=BANDLAB)
for sp in SP:
    i = IDX[sp]
    a.semilogx(k, PF[:, i].mean(0)/P_TH[i] - 1.0, color=COL[sp], marker="o",
               label=NAME[sp], **MK)
a.axhline(0.0, color="0.3", lw=1.0)
a.set_ylabel("$P(k)\\,/\\,P_{\\rm theory}(k) - 1$")
a.set_ylim(-0.15, 0.15)
a.set_title("Whole box against theory: the estimator is unbiased")
a.legend(framealpha=0.95, loc="lower left")
decorate_k(a)

i = IDX[ONE]
a = ax[1, 0]
a.loglog(k, P_TH[i], color="0.45", lw=4.0, alpha=0.28, solid_capstyle="round",
         label="linear theory")
a.loglog(k, P_WIN[i], color=PENC, lw=4.0, alpha=0.28, solid_capstyle="round",
         label=r"theory $\ast$ pencil window")
a.loglog(k, PF[:, i].mean(0), color=BOXC, marker="o", label="whole box", **MK)
a.loglog(k, PP[:, i].mean((0, 1)), color=PENC, ls="--", marker="s",
         label="pencil zoom-in region", **MK)
a.set_ylabel("$P(k)$  [$(\\mathrm{Mpc}/h)^3$]")
a.set_title(f"Whole box against pencil, {NAME[ONE]}")
a.legend(framealpha=0.95, loc="lower left")
decorate_k(a)

a = ax[1, 1]
a.fill_between(k, -sig, sig, color="0.75", alpha=0.45, lw=0, zorder=0, label=BANDLAB)
a.semilogx(k, PF[:, i].mean(0)/P_TH[i] - 1.0, color=BOXC, marker="o",
           label="whole box", **MK)
a.semilogx(k, PP[:, i].mean((0, 1))/P_TH[i] - 1.0, color=PENC, ls="--", marker="s",
           label="pencil zoom-in region", **MK)
a.semilogx(k, P_WIN[i]/P_TH[i] - 1.0, color="0.35", lw=1.8, ls=":",
           label=r"theory $\ast$ window, no realization")
a.axhline(0.0, color="0.3", lw=1.0)
a.set_ylabel("$P(k)\\,/\\,P_{\\rm theory}(k) - 1$")
a.set_ylim(-0.45, 0.20)
a.set_title("The pencil offset is geometric: the window predicts it")
a.legend(framealpha=0.95, loc="lower right")
decorate_k(a)

fig.suptitle("A pencil subvolume estimates the window-convolved spectrum\n" + SUB,
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(A.out_pk)
print(f"wrote {A.out_pk}")

# ------------------------------------------------------------------ xi -------
XI_TH = {sp: xi_theory(P_TH[IDX[sp]]) for sp in SP}
fin = r > 0
fig, ax = plt.subplots(2, 2, figsize=(13.6, 10.4))

a = ax[0, 0]
for sp in SP:
    i, xt = IDX[sp], XI_TH[sp]
    xb = XF[:, i].mean(0)
    a.loglog(r[fin & (xt > 0)], xt[fin & (xt > 0)], color=COL[sp], lw=4.0,
             alpha=0.22, solid_capstyle="round")
    a.loglog(r[fin & (xb > 0)], xb[fin & (xb > 0)], color=COL[sp], marker="o",
             label=NAME[sp], **MK)
a.set_ylabel(r"$\xi(r)$")
a.set_title("Whole box, each species, with its own theory beneath")
a.legend(framealpha=0.95)
decorate_r(a)

a = ax[0, 1]
for sp in SP:
    i, xt = IDX[sp], XI_TH[sp]
    g = fin & (xt > 0) & (r <= width)
    a.semilogx(r[g], (XF[:, i].mean(0)/xt - 1.0)[g], color=COL[sp], marker="o",
               label=NAME[sp], **MK)
a.axhline(0.0, color="0.3", lw=1.0)
a.set_ylabel(r"$\xi(r)\,/\,\xi_{\rm theory}(r) - 1$")
a.set_ylim(-0.15, 0.15)
a.set_title("Whole box against theory")
a.legend(framealpha=0.95, loc="lower left")
decorate_r(a)

i, xt = IDX[ONE], XI_TH[ONE]
xb, xp = XF[:, i].mean(0), XP[:, i].mean((0, 1))
a = ax[1, 0]
a.loglog(r[fin & (xt > 0)], xt[fin & (xt > 0)], color="0.45", lw=4.0, alpha=0.28,
         solid_capstyle="round", label="linear theory (one curve, not two)")
a.loglog(r[fin & (xb > 0)], xb[fin & (xb > 0)], color=BOXC, marker="o",
         label="whole box", **MK)
a.loglog(r[fin & (xp > 0)], xp[fin & (xp > 0)], color=PENC, ls="--", marker="s",
         label="pencil zoom-in region", **MK)
a.set_ylabel(r"$\xi(r)$")
a.set_title(f"Whole box against pencil, {NAME[ONE]}")
a.legend(framealpha=0.95, loc="lower left")
decorate_r(a)

a = ax[1, 1]
g = fin & (xt > 0) & (r <= width)
a.semilogx(r[g], (xb/xt - 1.0)[g], color=BOXC, marker="o", label="whole box", **MK)
a.semilogx(r[g], (xp/xt - 1.0)[g], color=PENC, ls="--", marker="s",
           label="pencil zoom-in region", **MK)
a.axhline(0.0, color="0.3", lw=1.0)
a.set_ylabel(r"$\xi(r)\,/\,\xi_{\rm theory}(r) - 1$")
a.set_ylim(-0.45, 0.20)
a.set_title("No offset: the mask cancels between the pair counts")
a.legend(framealpha=0.95, loc="lower left")
decorate_r(a)

fig.suptitle("The same subvolume estimates the correlation function\n"
             "with no window to convolve\n" + SUB, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(A.out_xi)
print(f"wrote {A.out_xi}\n")

lo = k <= 2*float(h5py.File(files[0]).attrs["dkperp"])
print(f"{'species':<14}{'P box/theory':>14}{'P pen/theory':>14}{'window/theory':>15}"
      f"{'xi box/theory':>15}{'xi pen/theory':>15}")
for sp in SP:
    i, xt = IDX[sp], XI_TH[sp]
    g = fin & (xt > 0) & (r <= width)
    w = np.abs(xt[g])
    print(f"{NAME[sp]:<14}{(PF[:, i].mean(0)/P_TH[i])[lo].mean():>14.4f}"
          f"{(PP[:, i].mean((0,1))/P_TH[i])[lo].mean():>14.4f}"
          f"{(P_WIN[i]/P_TH[i])[lo].mean():>15.4f}"
          f"{np.average((XF[:, i].mean(0)/xt)[g], weights=w):>15.4f}"
          f"{np.average((XP[:, i].mean((0,1))/xt)[g], weights=w):>15.4f}")
