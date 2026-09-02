#!/usr/bin/env python3
"""Fit an amplitude to each pencil spectrum, with and without the window in the model.

The question "does this pencil match theory?" is an inference question, and the
way to ask it is to fit a parameter rather than to search for a realization. Fit

    P_hat(k) = A * P_model(k) + noise,     Cov estimated from the ensemble,

by generalised least squares, once with P_model = theory convolved with the
pencil window, and once with P_model = the raw theory. The covariance is taken
from the scatter of the measurements themselves, so the correlations the window
induces between neighbouring k bins are included rather than assumed away.

What the two fits should give, if the window belongs in the model:

  * windowed:  A centred on 1, because that model is correct.
  * raw:       A centred well below 1, because the model overpredicts the data
               by the geometric deficit, and the fit absorbs it into A.

And the realizations a seed search keeps are those whose *raw* fit returns A = 1.
Read through the correct model, those same realizations have A well above 1:
they are not universes that match the theory, they are universes with too much
large-scale power.

Usage:
    python fit_amplitude.py --data DIR [--nchunk 60]
"""
import argparse, glob, os
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "big128"))
ap.add_argument("--nchunk", type=int, default=60, help="chunks to read")
ap.add_argument("--keep", type=float, default=0.01, help="fraction a search would keep")
ap.add_argument("--out", default=os.path.join(paths.FIGS, "amplitude_fit.png"))
A = ap.parse_args()

chunks = sorted(glob.glob(f"{A.data}/chunk_*.hdf5"))[:A.nchunk]
if not chunks:
    raise SystemExit(f"no chunk_*.hdf5 under {A.data}")
with h5py.File(chunks[0]) as f:
    nm = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
    SP = nm.index("matter")
    k, P_th, P_win = f["k"][:], f["P_theory"][SP], f["P_win"][SP]
    dkperp = f.attrs["dkperp"]
band = (k > 0) & (k <= 2*dkperp) & np.isfinite(P_win) & (P_win > 0)

P = np.concatenate([h5py.File(fn)["P_pencil"][:, SP][:, :, band].reshape(-1, band.sum())
                    for fn in chunks])
print(f"{len(P):,} pencil spectra, {band.sum()} k bins over "
      f"{k[band][0]:.3f}-{k[band][-1]:.3f} h/Mpc\n")

# Covariance of the measurements themselves. The window correlates neighbouring
# bins, so the off-diagonal terms are not optional.
Cov = np.cov(P.T)
Ci = np.linalg.pinv(Cov)


def gls(model):
    """Generalised least squares amplitude for every pencil against one model."""
    w = Ci @ model
    return (P @ w)/(model @ w)


fits = {"windowed model": gls(P_win[band]), "raw theory model": gls(P_th[band])}
crit = np.sqrt((np.log(P/P_th[band])**2).mean(1))          # the search's criterion
sel = np.argsort(crit)[:max(1, int(round(A.keep*len(crit))))]

print(f"{'model fitted':<20} {'mean A':>9} {'scatter':>9} {'A of the kept':>15}")
for name, a in fits.items():
    print(f"{name:<20} {a.mean():9.4f} {a.std():9.4f} {a[sel].mean():15.4f}")

aw, ar = fits["windowed model"], fits["raw theory model"]
print(f"\nThe search keeps pencils whose raw-model amplitude is near 1: "
      f"{ar[sel].mean():.3f} against {ar.mean():.3f} for all pencils.")
print(f"Those same pencils, read through the correct windowed model, have "
      f"A = {aw[sel].mean():.3f}, i.e. {100*(aw[sel].mean()-1):.0f}% too much "
      f"large-scale power.")
print(f"That is {(aw[sel].mean()-aw.mean())/aw.std():.2f} standard deviations "
      f"from a typical realization.")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
for ax, (name, a), col in ((a1, ("windowed model", aw), "C0"),
                           (a2, ("raw theory model", ar), "C1")):
    # Densities, not counts: the kept sample is 1% of the whole and would be a
    # flat line on a count axis, which is exactly what the figure has to show.
    bins = np.linspace(*np.percentile(np.concatenate([a, a[sel]]), [0.1, 99.9]), 70)
    ax.hist(a, bins=bins, color=col, alpha=0.7, edgecolor="none", density=True,
            label=f"all {len(a):,} regions")
    ax.hist(a[sel], bins=bins, histtype="step", lw=1.9, color="C3", density=True,
            label=f"kept by the search ({100*A.keep:g}%)")
    ax.axvline(1.0, color="0.2", lw=1.6, label="A = 1")
    ax.axvline(a.mean(), color=col, lw=1.4, ls="--", label=f"all: mean {a.mean():.3f}")
    ax.axvline(a[sel].mean(), color="C3", lw=1.4, ls=":",
               label=f"kept: mean {a[sel].mean():.3f}")
    ax.set_xlabel("$A$   (1 = the region has exactly the power the model predicts)")
    ax.set_ylabel("probability density")
    ax.legend(fontsize=8.5, framealpha=0.95, loc="upper left")

a1.set_title("$P_{\\rm model}$ = theory $\\ast$ window\n"
             "the curve a subvolume is unbiased for", fontsize=10.5)
a2.set_title("$P_{\\rm model}$ = raw linear theory\n"
             "the curve the seed search asks for", fontsize=10.5)

fig.suptitle("How much power does a subvolume have, compared with the model?\n"
             "Fit one number $A$ to each subvolume:  "
             "$\\hat P(k) = A \\, P_{\\rm model}(k)$\n"
             f"{len(P):,} subvolumes, generalised least squares over "
             f"k = {k[band][0]:.3f}-{k[band][-1]:.3f} $h$ Mpc$^{{-1}}$ "
             f"with the measured covariance", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(A.out)
print(f"\nwrote {A.out}")
