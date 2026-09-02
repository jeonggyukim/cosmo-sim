"""Validate the monofonIC LagrangianDensityOnly output.

Reads the delta(q) grid written by the `lagrangian-density` branch and the
companion 2LPT particle IC generated from the same seed, and compares both
against the back-scaled CLASS power spectrum monofonIC writes for the run.

Usage:  python plot_deltaq_check.py
"""
import numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
T = paths.ROOT
FDQ = f"{T}/n64_deltaq_z200_L500/delta_q_n64.hdf5"
FTH = f"{T}/n64_deltaq_z200_L500/deltaq_n64_input_powerspec.txt"
FIC = f"{T}/n64_2lpt_dm_z200_L500/ics_dm_n64.hdf5"
FPNG = f"{T}/n64_deltaq_z200_L500/delta_q_n64_pk.png"
hval = 0.6711

with h5py.File(FDQ) as f:
    d = f["delta_q"][:].astype(float)
    L, N = float(f["Header"].attrs["BoxSize"]), int(f["Header"].attrs["GridRes"])
    z, Dplus = float(f["Header"].attrs["zstart"]), float(f["Header"].attrs["Dplus"])
dx, kf, kny = L/N, 2*np.pi/L, np.pi*N/L

kax = np.fft.fftfreq(N, 1.0/N)*kf
kk = np.sqrt(kax[:,None,None]**2 + kax[None,:,None]**2 + kax[None,None,:]**2).ravel()
edges = np.arange(0.5, N/2+1.0)*kf
ib = np.digitize(kk, edges)
nb = len(edges)-1
kb = np.array([kk[ib==i+1].mean() for i in range(nb)])
binned = lambda P: np.array([P[ib==i+1].mean() for i in range(nb)])

# delta(q): defined on the grid, so no window and no shot noise to correct
Pq = binned((L**3*np.abs(np.fft.fftn(d)/N**3)**2).ravel())

# particle IC on the same bins: CIC + interlacing, CIC window deconvolved
def cic(pos):
    g = np.zeros((N, N, N))
    u = pos/dx
    i0 = np.floor(u).astype(int); w1 = u - i0; w0 = 1.0 - w1
    i0 %= N; i1 = (i0 + 1) % N
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                wa = (w0[:,0] if a==0 else w1[:,0])*(w0[:,1] if b==0 else w1[:,1])*(w0[:,2] if c==0 else w1[:,2])
                np.add.at(g, ((i0[:,0] if a==0 else i1[:,0]),
                              (i0[:,1] if b==0 else i1[:,1]),
                              (i0[:,2] if c==0 else i1[:,2])), wa)
    return g/g.mean() - 1.0

with h5py.File(FIC) as f:
    x = f["PartType1/Coordinates"][:].astype(float)*hval        # Mpc -> Mpc/h
ph = np.exp(0.5j*dx*(kax[:,None,None] + kax[None,:,None] + kax[None,None,:]))
dkp = 0.5*(np.fft.fftn(cic(x))/N**3 + (np.fft.fftn(cic((x + 0.5*dx) % L))/N**3)*ph)
sinc = lambda u: np.sinc(u/np.pi)
W = (sinc(kax[:,None,None]*dx/2)*sinc(kax[None,:,None]*dx/2)*sinc(kax[None,None,:]*dx/2))**2
Pp = binned((L**3*np.abs(dkp/W)**2).ravel())

# monofonIC writes [A(k) D+(zstart)]^2 in its internal amplitude units, which carry
# the (2pi)^-3/2 of volfac; the standard-convention P(k) is (2pi)^3 times that.
th = np.loadtxt(FTH)
kth, Pth = th[:,0], th[:,1]*(2*np.pi)**3
Pth_b = np.exp(np.interp(np.log(kb), np.log(kth), np.log(Pth)))
rq, rp = Pq/Pth_b, Pp/Pth_b

sel = kb <= 0.9*kny
print(f"N={N}^3  L={L:g} Mpc/h  z={z:g}  D+={Dplus:.6g}   rms delta(q) = {d.std():.6g}")
print(f"delta(q)/theory,  k<0.9k_Ny : median {np.median(rq[sel]):.4f}  max dev {100*np.abs(rq[sel]-1).max():.2f}%")
print(f"particles/theory, k<0.9k_Ny : median {np.median(rp[sel]):.4f}  max dev {100*np.abs(rp[sel]-1).max():.2f}%")

fig = plt.figure(figsize=(11.5, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[2.4, 1], hspace=0.06, wspace=0.40)
axs = fig.add_subplot(gs[:, 0]); ax0 = fig.add_subplot(gs[0, 1]); ax1 = fig.add_subplot(gs[1, 1], sharex=ax0)

v = np.abs(d).max()
im = axs.imshow(d[:, :, 0], origin="lower", extent=[0, L, 0, L], cmap="RdBu_r", vmin=-v, vmax=v)
axs.set_xlabel(r"$q_x\ [\mathrm{Mpc}/h]$"); axs.set_ylabel(r"$q_y\ [\mathrm{Mpc}/h]$")
axs.set_title(rf"$\delta(q)$ slice at $q_z=0$,   rms $= {d.std():.2e}$", fontsize=9)
fig.colorbar(im, ax=axs, fraction=0.046, pad=0.03, label=r"$\delta(q)$")

m = (kth >= 0.7*kb[0]) & (kth <= 1.4*kny)
ax0.loglog(kth[m], Pth[m], "-", color="0.35", lw=1.4,
           label=r"theory: $(2\pi)^3\times$ monofonIC $P_{\rm dtot}(k,a{=}a_{\rm start})$")
ax0.loglog(kb, Pq, "o", ms=4.5, color="C0", label=r"$\delta(q)$ grid (no window, no shot noise)")
ax0.loglog(kb, Pp, "^", ms=4.5, mfc="none", color="C3",
           label="2LPT particle IC, same seed (CIC + interlacing, deconvolved)")
ax0.axvline(kny, ls=":", color="0.6")
ax0.text(kny*0.96, Pth[m].max()*0.25, r"$k_{\rm Ny}$", ha="right", color="0.45", fontsize=9)
ax0.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$")
ax0.legend(frameon=False, fontsize=7.5, loc="lower left")
plt.setp(ax0.get_xticklabels(), visible=False)

ax1.semilogx(kb, rq, "o-", ms=4, color="C0", lw=1)
ax1.semilogx(kb, rp, "^--", ms=4, mfc="none", color="C3", lw=1)
ax1.axhline(1.0, color="0.35", lw=1)
ax1.axhspan(0.99, 1.01, color="0.85", zorder=0)
ax1.axvline(kny, ls=":", color="0.6")
ax1.set_ylim(0.94, 1.26)
ax1.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$")
ax1.set_ylabel(r"$P/P_{\rm th}$", fontsize=9)

fig.suptitle(r"monofonIC $\tt LagrangianDensityOnly$: linear Lagrangian density $\delta(q)$ and its $P(k)$"
             "\n"
             rf"$N={N}^3$, $L={L:g}\,$Mpc$/h$, $z_{{\rm start}}={z:g}$, $D_+={Dplus:.4g}$, "
             r"seed 12345, $\tt DoFixing=yes$, CV_22 cosmology; companion particle IC is 2LPT, DM only",
             fontsize=10)
fig.savefig(FPNG, dpi=150, bbox_inches="tight")
print("saved", FPNG)
