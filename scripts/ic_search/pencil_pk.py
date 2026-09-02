"""P(k) of a pencil subregion, measured against the window-convolved theory.

Geometry: the subregion spans 1/8 of the box in x and y and the full box in z.
The mask is therefore separable, W = W_x W_y W_z with W_z = 1 everywhere, so the
line-of-sight axis carries no window at all and only k_x, k_y are convolved.

Estimator, with W the 0/1 indicator of the pencil and f = <W^2> its volume
fraction (Park et al. 1994, eqs. 9-13, with w_j = 1 and no shot noise since the
field is a continuum grid):

    F(k)      = (1/N^3) sum_x W_x delta_x exp(-i k.x)
    P_meas(k) = V |F(k)|^2 / f

The same window applied to the theory spectrum gives what the measurement should
reproduce, a circular convolution over the k-grid:

    P_win(k)  = sum_k' |Whatt(k-k')|^2 P_th(k') / f,   Whatt(k) = (1/N^3) sum_x W_x e^{-i k.x}

with sum_k |Whatt(k)|^2 = f by Parseval. No deconvolution is attempted: the
comparison is made in the observed basis, which is always well posed.
"""
import numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse, glob, os
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="n64_deltaq_z200_L700",
                help="run directory under $MONOFONIC_TESTS holding delta_q*.hdf5 "
                     "and monofonIC's *_input_powerspec.txt")
ap.add_argument("--png", default=None)
ap.add_argument("--frac", type=int, default=8, help="pencil is 1/frac of the box in two axes")
ap.add_argument("--label", default="", help="extra text for the figure title, e.g. DoFixing=no")
A = ap.parse_args()

T = paths.ROOT
RUN = A.run
RUNDIR = os.path.join(T, RUN)
def one(pattern):
    hits = sorted(glob.glob(os.path.join(RUNDIR, pattern)))
    if not hits:
        raise SystemExit(f"no {pattern} in {RUNDIR}")
    return hits[0]
FDQ, FTH = one("delta_q*.hdf5"), one("*_input_powerspec.txt")
FPNG = A.png or os.path.join(paths.FIGS, f"pencil_pk_{RUN}.png")
FRAC = A.frac                              # pencil is 1/FRAC of the box in two axes

with h5py.File(FDQ) as f:
    d = f["delta_q"][:].astype(float)
    L, N = float(f["Header"].attrs["BoxSize"]), int(f["Header"].attrs["GridRes"])
    z = float(f["Header"].attrs["zstart"])
V, kf, kny = L**3, 2*np.pi/L, np.pi*N/L
npen = N//FRAC                              # cells across the pencil
lperp = L/FRAC                              # transverse size, Mpc/h
dkperp = 2*np.pi/lperp                      # window width in k_perp

kax = np.fft.fftfreq(N, 1.0/N)*kf
KX, KY, KZ = np.meshgrid(kax, kax, kax, indexing="ij")
kk = np.sqrt(KX**2 + KY**2 + KZ**2)
edges = np.arange(0.5, N/2+1.0)*kf
ib = np.digitize(kk.ravel(), edges); nb = len(edges)-1
kb = np.array([kk.ravel()[ib==i+1].mean() for i in range(nb)])
binned = lambda P: np.array([P.ravel()[ib==i+1].mean() for i in range(nb)])

# theory on the 3-d k grid, from monofonIC's back-scaled table (times (2 pi)^3)
th = np.loadtxt(FTH); kth, Pth1 = th[:,0], th[:,1]*(2*np.pi)**3
Pth = np.zeros_like(kk)
good = kk > 0
Pth[good] = np.exp(np.interp(np.log(kk[good]), np.log(kth), np.log(Pth1)))

# pencil mask and its window power
W = np.zeros((N, N, N)); W[:npen, :npen, :] = 1.0
fvol = W.mean()
Wk2 = np.abs(np.fft.fftn(W)/N**3)**2
print(f"pencil {npen}x{npen}x{N} cells = {lperp:.1f} x {lperp:.1f} x {L:.0f} Mpc/h")
print(f"volume fraction f = {fvol:.6f} (1/{1/fvol:.0f}),  sum|W_k|^2 = {Wk2.sum():.6f}  [must equal f]")
print(f"transverse window width dk_perp = {dkperp:.4f} h/Mpc = {dkperp/kf:.0f} k_fund;  k_Ny = {kny:.4f}")

# window-convolved theory: circular convolution over the k grid
Pwin = np.real(np.fft.ifftn(np.fft.fftn(Pth)*np.fft.fftn(Wk2)))/fvol

# full-box measurement, for reference
Pfull = binned(V*np.abs(np.fft.fftn(d)/N**3)**2)

# One pencil, and the means over all disjoint pencils. The box is periodic and
# the field statistically isotropic, so pencils along x and y are as valid as
# along z: 3 orientations x FRAC^2 transverse positions. The three window
# functions are rotations of one another, so after binning in |k| they share one
# window-convolved theory curve. The 3 orientations tile the same modes, so they
# are not independent and the scatter falls by less than sqrt(3).
def pencil_mask(axis, i, j):
    W = np.zeros((N, N, N))
    sl = [slice(i*npen, (i+1)*npen), slice(j*npen, (j+1)*npen)]
    idx = [None, None, None]
    idx[axis] = slice(None)
    rest = [a for a in range(3) if a != axis]
    idx[rest[0]], idx[rest[1]] = sl[0], sl[1]
    W[tuple(idx)] = 1.0
    return W

def pencil_pk(axis, i, j):
    return V*np.abs(np.fft.fftn(pencil_mask(axis, i, j)*d)/N**3)**2/fvol

P1 = pencil_pk(2, 0, 0)
Pax = [np.mean([pencil_pk(a, i, j) for i in range(FRAC) for j in range(FRAC)], axis=0)
       for a in range(3)]
Pz = Pax[2]
Pall = np.mean(Pax, axis=0)
NPEN = 3*FRAC*FRAC

b_one, b_all, b_win, b_th = binned(P1), binned(Pall), binned(Pwin), binned(Pth)
b_z = binned(Pz)

# The convolved theory is a smooth function of k; resolve it with fine log bins
# (the modes are the same, only the binning is finer), and take the unconvolved
# theory straight from the CLASS table, which is sampled every 1% in k.
fedges = np.logspace(np.log10(0.9*kf), np.log10(1.02*kk.max()), 160)
fib = np.digitize(kk.ravel(), fedges)
occ = np.array([(fib==i+1).sum() for i in range(len(fedges)-1)])
keep = occ > 0
kfine = np.array([kk.ravel()[fib==i+1].mean() for i in range(len(fedges)-1) if occ[i] > 0])
wfine = np.array([Pwin.ravel()[fib==i+1].mean() for i in range(len(fedges)-1) if occ[i] > 0])
print(f"fine theory curve: {len(kfine)} points (coarse bins: {nb})")

# line-of-sight modes: the k_perp = 0 column, where the pencil retains full resolution
kz = kax[:N//2]
# Each orientation has its own line of sight, so take each mean along its own
# axis: the k_perp = 0 column of the z-pencils is a transverse-mixed column for
# the x- and y-pencils, not their line of sight.
los_cols = [Pax[0][:N//2, 0, 0], Pax[1][0, :N//2, 0], Pax[2][0, 0, :N//2]]
los_all = np.mean(los_cols, axis=0)
los_win, los_th = Pwin[0, 0, :N//2], Pth[0, 0, :N//2]

sel = (kb > 2*dkperp) & (kb <= 0.9*kny)
r = b_all[sel]/b_win[sel]
rz = b_z[sel]/b_win[sel]
print(f"\nmean of {FRAC*FRAC} z-pencils  / window-convolved theory, k > 2 dk_perp: "
      f"median {np.median(rz):.4f}, rms dev {100*np.std(rz-1):.2f}%, max dev {100*np.abs(rz-1).max():.1f}%")
print(f"mean of {NPEN} pencils (x,y,z) / window-convolved theory, k > 2 dk_perp: "
      f"median {np.median(r):.4f}, rms dev {100*np.std(r-1):.2f}%, max dev {100*np.abs(r-1).max():.1f}%")
r2 = b_all/b_th
print(f"same, against the UNconvolved theory                        : "
      f"median {np.median(r2[sel]):.4f}, and {r2[0]:.3f} in the first bin")

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6))
fig.subplots_adjust(wspace=0.34)

a = ax[0]
mth = (kth >= 0.85*kb[0]) & (kth <= 1.15*kny)
a.loglog(kth[mth], Pth1[mth], "-", color="0.3", lw=1.5, label=r"theory $P(k)$ (full box)")
a.loglog(kfine, wfine, "-", color="C1", lw=1.8, label=r"theory $\ast$ pencil window")
a.loglog(kb, b_one, "o", ms=4, mfc="none", color="C0", label="one pencil")
a.loglog(kb, b_all, "s", ms=4.5, color="C3", label=rf"mean of {NPEN} disjoint pencils ($x,y,z$)")
a.axvline(dkperp, ls="--", color="C1", lw=1)
a.text(dkperp*1.08, b_th.max()*0.5, r"$\Delta k_\perp = 2\pi/\ell_\perp$", color="C1", fontsize=8)
a.axvline(kny, ls=":", color="0.6")
a.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); a.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$")
a.set_title("(a)  pencil $P(k)$ vs window-convolved theory", fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="lower left")

a = ax[1]
a.semilogx(kb, b_z/b_win, "o-", ms=3.5, lw=1, mfc="none", color="C0",
           label=rf"{FRAC*FRAC} $z$-pencils $/$ (theory $\ast$ window)")
a.semilogx(kb, b_all/b_win, "s-", ms=4, lw=1, color="C3",
           label=rf"{NPEN} pencils, all axes $/$ (theory $\ast$ window)")
a.semilogx(kb, b_all/b_th, "^--", ms=4, mfc="none", lw=1, color="0.45", label=r"mean of pencils $/$ theory")
a.axhline(1.0, color="0.3", lw=1); a.axhspan(0.95, 1.05, color="0.88", zorder=0)
a.axvline(dkperp, ls="--", color="C1", lw=1); a.axvline(kny, ls=":", color="0.6")
a.set_ylim(0.5, 1.6)
a.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); a.set_ylabel("ratio")
a.set_title("(b)  the window is the whole difference", fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="lower right")

a = ax[2]
mth2 = (kth >= 0.85*kz[1]) & (kth <= 1.15*kny)
a.loglog(kth[mth2], Pth1[mth2], "-", color="0.3", lw=1.5, label=r"theory $P(k_\parallel)$")
a.loglog(kz[1:], los_win[1:], "-", color="C1", lw=1.8, label=r"theory $\ast$ window, $k_\perp=0$")
a.loglog(kz[1:], los_all[1:], "s", ms=4.5, color="C3", label=rf"{NPEN} pencils, each along its own axis")
a.axvline(kny, ls=":", color="0.6")
a.set_xlabel(r"$k_\parallel\ [h/\mathrm{Mpc}]$"); a.set_ylabel(r"$P\ [(\mathrm{Mpc}/h)^3]$")
a.set_title(r"(c)  line-of-sight modes: $k_\perp$ still folds in", fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="lower left")

fig.suptitle(r"$P(k)$ of a pencil subregion by masked FFT: "
             rf"${npen}\times{npen}\times{N}$ cells $= {lperp:.0f}\times{lperp:.0f}\times{L:.0f}$ Mpc$/h$, "
             r"cut in two axes, periodic in the third"
             "\n"
             rf"parent field: monofonIC $\delta_{{\rm m}}(q)$, $N={N}^3$, $L={L:g}\,$Mpc$/h$, $z={z:g}$, "
             rf"seed 12345, {A.label or r'$\tt DoFixing=yes$'}, CV_22 cosmology",
             fontsize=10.5, y=1.02)
fig.savefig(FPNG, dpi=300, bbox_inches="tight")
print("saved", FPNG)
