#!/usr/bin/env python3
"""What the sweep actually measures, shown as slices through one realization.

Every number in the seed-search analysis comes from the chain drawn here, in
order: the Lagrangian density field, the same field smoothed on a scale R, the
tidal tensor built from it, the collapse geometry that tensor implies, and the
displacement it drives. The pencil subvolume is outlined on every panel, so what
is a property of the region and what is a property of the box it sits in can be
told apart by eye.

The definitions, in the order the panels use them. delta(q) is the linear density
contrast on the Lagrangian grid. Smoothing is a Gaussian of radius R in
configuration space, delta_R(k) = delta(k) exp(-k^2 R^2 / 2). The tidal tensor is
the Hessian of the potential that sources delta_R,

    T_ij(k) = k_i k_j delta_R(k) / k^2,

so its trace is delta_R itself and its traceless part is the shear
s_ij = T_ij - delta_ij delta_R/3, whose square is

    s^2 = sum_ij T_ij^2 - delta_R^2/3,

equal to (2/3)<delta_R^2> on average for a Gaussian field. The T-web class counts
how many eigenvalues of T_ij are positive, which is how many axes are collapsing:
3 a knot, 2 a filament, 1 a sheet, 0 a void (Hahn et al. 2007). The Zel'dovich
displacement is Psi(k) = i k delta(k)/k^2, and its mean over the region is the
bulk flow the sweep records.

The slice is taken perpendicular to the pencil's long axis, so the region appears
as a square; along the third axis it spans the whole box.

Usage:
    python plot_field_slices.py --field DELTAQ.hdf5 [--frac 8] [--out PNG]
"""
import argparse, os
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--field", default=os.path.join(
    paths.ROOT, "n64_deltaq_z200_L700", "delta_q_n64_L700.hdf5"))
ap.add_argument("--dataset", default=None,
                help="dataset holding the density; default is the first delta_q* found")
ap.add_argument("--frac", type=int, default=8,
                help="the pencil is 1/frac of the box in the two transverse axes")
ap.add_argument("--smooth-frac", type=float, nargs="+", default=[0.15, 0.25, 0.5],
                help="smoothing radii as fractions of the pencil width")
ap.add_argument("--slice", type=int, default=None,
                help="index of the slice along the pencil's long axis; default mid-box")
ap.add_argument("--out", default=os.path.join(paths.FIGS, "field_slices.png"))
A = ap.parse_args()

with h5py.File(A.field) as f:
    name = A.dataset or next(k for k in f if k.startswith("delta_q"))
    d = f[name][:].astype(float)
    hdr = f["Header"].attrs if "Header" in f else {}
    L = float(hdr.get("BoxSize", 700.0))
    N = int(hdr.get("GridRes", d.shape[0]))
    z = float(hdr.get("zstart", 200.0))

npen = N//A.frac
width = L/A.frac
RS = [r*width for r in A.smooth_frac]
kf = 2*np.pi/L
ka = np.fft.fftfreq(N, 1.0/N)*kf
KX, KY, KZ = np.meshgrid(ka, ka, ka, indexing="ij")
KV = [KX, KY, KZ]
kk2 = KX**2 + KY**2 + KZ**2
K2 = kk2.copy(); K2[0, 0, 0] = 1.0        # guarded only where 1/k^2 is taken
dk = np.fft.fftn(d)

# The slice runs perpendicular to the pencil's long axis, which is taken as z, so
# the region is the square [0, npen) x [0, npen) in the plane drawn.
iz = A.slice if A.slice is not None else N//2
sl = (slice(None), slice(None), iz)
ext = [0, L, 0, L]


def smooth(R):
    return np.real(np.fft.ifftn(dk*np.exp(-0.5*kk2*R**2)))


def tidal(R):
    """T_ij on the slice, and delta_R on the slice."""
    dks = dk*np.exp(-0.5*kk2*R**2)
    dsm = np.real(np.fft.ifftn(dks))
    T = np.empty((N, N, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            full = np.real(np.fft.ifftn(KV[a]*KV[b]/K2*dks))
            T[..., a, b] = T[..., b, a] = full[sl]
    return dsm, T


psi = [np.real(np.fft.ifftn(1j*KV[a]/K2*dk)) for a in range(3)]

# The three radii are the ones the sweep smooths on, as fractions of the pencil
# width, so the panels show the same scales the recorded numbers come from.
R1, R2, R3 = RS[0], RS[len(RS)//2], RS[-1]
d1, T1 = tidal(R1)
d3, T3 = tidal(R3)
d2 = smooth(R2)
s2 = (T1*T1).sum((-1, -2)) - d1[sl]**2/3.0
ev = np.linalg.eigvalsh(T1)                     # ascending
npos = (ev > 0).sum(-1)                         # 3 knot, 2 filament, 1 sheet, 0 void
npos3 = (np.linalg.eigvalsh(T3) > 0).sum(-1)

plt.rcParams.update({"font.size": 11.5, "axes.titlesize": 12,
                     "axes.labelsize": 11.5, "xtick.labelsize": 10,
                     "ytick.labelsize": 10})
fig, ax = plt.subplots(2, 4, figsize=(19.6, 10.0))


_c0 = (N - npen)//2
PEN = (slice(_c0, _c0 + npen), slice(_c0, _c0 + npen))


def frame(a, title, note=None, label=False):
    """Outline a pencil, and quote what the sweep records from this panel.

    The square is drawn at the centre of the box and is there to give the scale
    of a region against the field, not to mark which region was measured: every
    pencil has the same shape and the sweep uses all of them. It is named once,
    on the first panel, rather than repeated on all eight.

    The scalar the sweep stores is an average over a region of this size, so
    printing the region value beside the whole-box value turns each panel into
    the definition of the number rather than an illustration of it.
    """
    c = 0.5*(L - width)
    a.add_patch(Rectangle((c, c), width, width, fill=False, ec="k", lw=2.0))
    if label:
        a.text(c + 0.5*width, c + width*1.02, "pencil", ha="center", va="bottom",
               fontsize=10, color="k")
    a.set_title(title)
    a.set_xticks([0, L/2, L]); a.set_yticks([0, L/2, L])
    if note:
        a.text(0.985, 0.015, note, transform=a.transAxes, ha="right", va="bottom",
               fontsize=9.5, color="k",
               bbox=dict(fc="white", ec="0.7", alpha=0.88, pad=3.0))


def show(a, img, title, cmap="RdBu_r", sym=True, note=None, label=False, **kw):
    v = np.max(np.abs(img)) if sym else None
    im = a.imshow(img.T, origin="lower", extent=ext, cmap=cmap,
                  vmin=-v if sym else None, vmax=v if sym else None, **kw)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    frame(a, title, note, label)
    return im


def pair(img, fmt="{:+.4f}"):
    """The value over the pencil and over the whole slice, as one label."""
    return ("$\\bar{Q}$ pencil " + fmt.format(img[PEN].mean())
            + "\nbox " + fmt.format(img.mean()))


show(ax[0, 0], d[sl], r"(a)  $\delta(q)$, unsmoothed", note=pair(d[sl]),
     label=True)
show(ax[0, 1], d1[sl], rf"(b)  $\delta_R$, $R$ = {R1:.0f} Mpc/$h$", note=pair(d1[sl]))
show(ax[0, 2], d2[sl], rf"(c)  $\delta_R$, $R$ = {R2:.0f} Mpc/$h$", note=pair(d2[sl]))
show(ax[0, 3], d3[sl], rf"(d)  $\delta_R$, $R$ = {R3:.0f} Mpc/$h$", note=pair(d3[sl]))

srms = np.sqrt(s2)
im = ax[1, 0].imshow(srms.T, origin="lower", extent=ext, cmap="magma")
fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.03)
frame(ax[1, 0], rf"(e)  tidal shear $\sqrt{{s^2}}$, $R$ = {R1:.0f} Mpc/$h$",
      "$\\langle s^2\\rangle^{1/2}$ pencil "
      f"{np.sqrt(s2[PEN].mean()):.4f}" + "\nbox " + f"{np.sqrt(s2.mean()):.4f}")

WEB = ListedColormap(["#3b4cc0", "#8db0d5", "#f2b134", "#b40426"])
im = ax[1, 1].imshow(npos.T, origin="lower", extent=ext, cmap=WEB,
                     norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4))
cb = fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.03, ticks=[0, 1, 2, 3])
cb.ax.set_yticklabels(["void", "sheet", "filament", "knot"])
# The knot and void fractions the sweep records are the areas of the two extreme
# classes inside the outlined square, so they are read straight off this panel.
fp = np.bincount(npos[PEN].ravel(), minlength=4)/npos[PEN].size
fb = np.bincount(npos.ravel(), minlength=4)/npos.size
frame(ax[1, 1], rf"(f)  T-web class, $R$ = {R1:.0f} Mpc/$h$",
      f"knot  pencil {fp[3]:.3f}  box {fb[3]:.3f}\n"
      f"void  pencil {fp[0]:.3f}  box {fb[0]:.3f}")

pmag = np.sqrt(sum(psi[a][sl]**2 for a in range(3)))
im = ax[1, 3].imshow(pmag.T, origin="lower", extent=ext, cmap="viridis")
fig.colorbar(im, ax=ax[1, 3], fraction=0.046, pad=0.03)
st = max(1, N//24)
g = np.arange(0, N, st)*(L/N) + 0.5*L/N
ax[1, 3].quiver(g, g, psi[0][sl][::st, ::st].T, psi[1][sl][::st, ::st].T,
                color="white", scale_units="xy", angles="xy", width=0.004,
                alpha=0.85)
# The same classification on the largest radius. At half the pencil width the
# map is smooth on the scale of the region itself, which is why a class fraction
# measured there describes the surroundings more than the region.
im3 = ax[1, 2].imshow(npos3.T, origin="lower", extent=ext, cmap=WEB,
                      norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4))
cb3 = fig.colorbar(im3, ax=ax[1, 2], fraction=0.046, pad=0.03, ticks=[0, 1, 2, 3])
cb3.ax.set_yticklabels(["void", "sheet", "filament", "knot"])
f3p = np.bincount(npos3[PEN].ravel(), minlength=4)/npos3[PEN].size
f3b = np.bincount(npos3.ravel(), minlength=4)/npos3.size
frame(ax[1, 2], rf"(g)  T-web class, $R$ = {R3:.0f} Mpc/$h$",
      f"knot  pencil {f3p[3]:.3f}  box {f3b[3]:.3f}\n"
      f"void  pencil {f3p[0]:.3f}  box {f3b[0]:.3f}")

bulk = np.array([psi[a][sl][PEN].mean() for a in range(3)])
frame(ax[1, 3], r"(h)  Zel'dovich displacement $|\Psi|$, in-plane arrows",
      f"bulk flow, pencil {np.linalg.norm(bulk):.2f} Mpc/$h$")

for a in ax.ravel():
    a.set_xlabel("Mpc/$h$")
    a.set_ylabel("Mpc/$h$")

fig.suptitle("The measurement chain, on one realization\n"
             f"$N={N}^3$, $L={L:g}$ Mpc/$h$, $z={z:g}$, pencil {width:g} Mpc/$h$ "
             f"across and spanning the box along the third axis; "
             f"slice at $z$ index {iz}", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(A.out)
print(f"wrote {A.out}\n")
print(f"field {A.field}, dataset {name}")
print(f"  <delta^2>            {d.var():.4e}")
# Both sides on the same slice: s2 is only ever formed there, and comparing it
# against the variance of the full 3-d field mixes two different samples.
print(f"  <s^2>/<delta_R^2>    {s2.mean()/d1[sl].var():.4f}   (2/3 expected)")
cnt = np.bincount(npos.ravel(), minlength=4)/npos.size
print(f"  web fractions on this slice: void {cnt[0]:.3f}, sheet {cnt[1]:.3f}, "
      f"filament {cnt[2]:.3f}, knot {cnt[3]:.3f}")
