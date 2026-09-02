"""Per-species check of the monofonIC LagrangianDensityOnly output.

Panels:
  (0,0) P(k) of delta_m(q), delta_c(q), delta_b(q) against the back-scaled CLASS
        spectra monofonIC writes for the run (columns P_dtot, P_dcdm, P_dbar of
        *_input_powerspec.txt, times (2 pi)^3), plus the 2LPT particle IC.
  (0,1) P_b/P_c, the baryon suppression at fixed seed.
  (1,0) measured / theory for each species.
  (1,1) P(k) of the relative mode delta_bc = delta_b - delta_c, as carried by the
        back-scaled grids and as carried by the particle masses (un-back-scaled).
  free slot reserved for a further panel.
"""
import numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from icpipe.field import deposit
from icpipe.windows import assignment_window_squared

import paths
T = paths.ROOT
RUN_DQ, RUN_IC = "n64_deltaq_z200_L700", "n64_2lpt_dmgas_z200_L700"
FDQ = f"{T}/{RUN_DQ}/delta_q_n64_L700.hdf5"
FTH = f"{T}/{RUN_DQ}/deltaq_n64_L700_input_powerspec.txt"
FIC = f"{T}/{RUN_IC}/ics_dmgas_n64_L700.hdf5"
FPNG = f"{T}/{RUN_DQ}/delta_q_n64_L700_species_pk.png"
hval, ORDER, DPI = 0.6711, 4, 300          # PCS assignment
FB = 0.049/0.3                             # Omega_b / Omega_m

with h5py.File(FDQ) as f:
    L, N = float(f["Header"].attrs["BoxSize"]), int(f["Header"].attrs["GridRes"])
    z, Dplus = float(f["Header"].attrs["zstart"]), float(f["Header"].attrs["Dplus"])
    dq = {s: f[d][:].astype(float) for s, d in
          [("m", "delta_q"), ("c", "delta_q_cdm"), ("b", "delta_q_baryon")]}
dx, kf, kny = L/N, 2*np.pi/L, np.pi*N/L

kax = np.fft.fftfreq(N, 1.0/N)*kf
KX, KY, KZ = np.meshgrid(kax, kax, kax, indexing="ij")
kk = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()
edges = np.arange(0.5, N/2+1.0)*kf
ib = np.digitize(kk, edges); nb = len(edges)-1
kb = np.array([kk[ib==i+1].mean() for i in range(nb)])
binned = lambda P: np.array([P.ravel()[ib==i+1].mean() for i in range(nb)])
F = lambda g: np.fft.fftn(g)/N**3
Pk = lambda g: binned(L**3*np.abs(F(g))**2)

Pg = {s: Pk(d) for s, d in dq.items()}
P_bc_bs = Pk(dq["b"] - dq["c"])                        # back-scaled relative mode

# particle IC. Positions carry the back-scaled total matter for BOTH species;
# the baryon-CDM split rides in the masses as the un-back-scaled delta_bc.
W2 = assignment_window_squared(KX, KY, KZ, kny, ORDER); W2[0,0,0] = 1.0
ph = np.exp(0.5j*dx*(KX + KY + KZ))
with h5py.File(FIC) as f:
    xd = f["PartType1/Coordinates"][:].astype(float)*hval        # Mpc -> Mpc/h
    xg = f["PartType0/Coordinates"][:].astype(float)*hval
    mg = f["PartType0/Masses"][:].astype(float)
dep = lambda p: (lambda g: g/g.mean() - 1.0)(deposit(p, L, N, ORDER))
dk = 0.5*(F(dep(xd)) + F(dep((xd + 0.5*dx) % L))*ph)
Pp_dm = binned(L**3*np.abs(dk)**2/W2)

# delta_bc from the gas masses: cell mean of the per-particle mass fluctuation,
# divided by C_gas = 1 - f_b. The displacement is <1% of a cell at z=200, so the
# Eulerian cell mean is the Lagrangian field to that accuracy.
mu = mg/mg.mean() - 1.0
# The cell mean is the mu field convolved with the assignment kernel, so its power
# carries one factor of |W(k)|^2 and has to be deconvolved like any deposit.
dbc_true = (deposit(xg, L, N, ORDER, weights=mu)/np.maximum(deposit(xg, L, N, ORDER), 1e-30))/(1.0 - FB)
P_bc_true = binned(L**3*np.abs(F(dbc_true))**2/W2)

th = np.loadtxt(FTH)
kth = th[:,0]
Pth = {"m": th[:,1]*(2*np.pi)**3, "c": th[:,2]*(2*np.pi)**3, "b": th[:,3]*(2*np.pi)**3}
ip = lambda P: np.exp(np.interp(np.log(kb), np.log(kth), np.log(P)))
Pth_b = {s: ip(P) for s, P in Pth.items()}

sel = kb <= 0.9*kny
for s in ("m", "c", "b"):
    r = Pg[s]/Pth_b[s]
    print(f"delta_{s}(q)/theory, k<0.9k_Ny : median {np.median(r[sel]):.5f}  max dev {100*np.abs(r[sel]-1).max():.2f}%")
rb_m, rb_t = Pg["b"]/Pg["c"], Pth_b["b"]/Pth_b["c"]
print(f"P_b/P_c at k_Ny : delta(q) {rb_m[-1]:.5f}, theory {rb_t[-1]:.5f}")
rr = P_bc_bs/(Dplus**2*P_bc_true)
print(f"rms delta_bc from per-particle masses (no smoothing): {(mu/(1.0-FB)).std():.4e}")
print(f"rms delta_bc from back-scaled grids                 : {(dq['b']-dq['c']).std():.4e}")
print(f"their ratio {(dq['b']-dq['c']).std()/(mu/(1.0-FB)).std():.6f}  vs  D+ = {Dplus:.6f}")
print(f"P_bc(back-scaled) / [D+^2 P_bc(masses)] : median {np.median(rr[sel]):.4f} over k<0.9k_Ny")

lab = {"m": r"$\delta_{\rm m}(q)$, total matter",
       "c": r"$\delta_{\rm c}(q)$, CDM",
       "b": r"$\delta_{\rm b}(q)$, baryons"}
col = {"m": "0.3", "c": "C0", "b": "C3"}
mrk = {"m": "o", "c": "s", "b": "D"}

fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.2))
fig.subplots_adjust(hspace=0.28, wspace=0.26)
a00, a01, a10, a11 = ax[0,0], ax[0,1], ax[1,0], ax[1,1]

m = (kth >= 0.7*kb[0]) & (kth <= 1.2*kny)
a00.loglog(kth[m], Pth["m"][m], "-", color="0.3", lw=1.2, label="theory (CLASS, back-scaled)")
for s in ("m", "c", "b"):
    a00.loglog(kb, Pg[s], mrk[s], ms=4.5, mfc="none" if s != "m" else col[s], color=col[s], label=lab[s])
a00.loglog(kb, Pp_dm, "+", ms=6, color="C2", label="2LPT particle IC, positions only")
a00.axvline(kny, ls=":", color="0.6")
a00.text(kny*0.97, Pth["m"][m].max()*0.4, r"$k_{\rm Ny}$", ha="right", color="0.45", fontsize=9)
a00.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); a00.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$")
a00.set_title(r"(a)  Lagrangian density $P(k)$ per species", fontsize=9.5)
a00.legend(frameon=False, fontsize=8, loc="lower left")

a01.semilogx(kth[m], Pth["b"][m]/Pth["c"][m], "-", color="0.3", lw=1.6,
             label=r"theory: $P_{\rm dbar}/P_{\rm dcdm}$")
a01.semilogx(kb, rb_m, "D", ms=4.5, mfc="none", color="C3", label=r"measured: $P_{\delta_{\rm b}}/P_{\delta_{\rm c}}$")
a01.axvline(kny, ls=":", color="0.6"); a01.axhline(1.0, color="0.8", lw=0.8)
a01.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); a01.set_ylabel(r"$P_{\delta_{\rm b}}(k)\,/\,P_{\delta_{\rm c}}(k)$")
a01.set_title(r"(b)  baryon suppression relative to CDM, same seed", fontsize=9.5)
a01.legend(frameon=False, fontsize=8, loc="lower left")

for s in ("m", "c", "b"):
    a10.semilogx(kb, Pg[s]/Pth_b[s], mrk[s]+"-", ms=3.5, lw=1, mfc="none", color=col[s], label=lab[s])
a10.axhline(1.0, color="0.35", lw=1); a10.axhspan(0.99, 1.01, color="0.88", zorder=0)
a10.axvline(kny, ls=":", color="0.6")
a10.set_ylim(0.97, 1.03)
a10.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); a10.set_ylabel(r"$P_{\rm measured}/P_{\rm theory}$")
a10.set_title(r"(c)  measured $/$ theory, per species  (grey band $\pm1\%$)", fontsize=9.5)
a10.legend(frameon=False, fontsize=8, loc="lower left")

a11.loglog(kb, P_bc_true, "o", ms=4.5, mfc="none", color="C4",
           label=r"$\delta_{bc}$ from the particle masses (un-back-scaled)")
a11.loglog(kb, P_bc_bs, "s", ms=4.5, mfc="none", color="C1",
           label=r"$\delta_{\rm b}(q)-\delta_{\rm c}(q)$ from the back-scaled grids")
a11.loglog(kb, P_bc_true*Dplus**2, "-", color="0.3", lw=1.2,
           label=r"$D_+^2 \times$ particle-mass $\delta_{bc}$")
a11.loglog(kb, Pg["m"], ":", color="0.6", lw=1.2, label=r"$\delta_{\rm m}(q)$, for scale")
a11.axvline(kny, ls=":", color="0.6")
a11.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); a11.set_ylabel(r"$P_{\delta_{bc}}(k)\ [(\mathrm{Mpc}/h)^3]$")
a11.set_title(r"(d)  the relative mode $\delta_{bc}=\delta_{\rm b}-\delta_{\rm c}$", fontsize=9.5)
a11.legend(frameon=False, fontsize=7.5, loc="lower left")
a11.text(0.33, 0.55,
         r"$\delta_{bc}$ has no gravitational source, so it stays"
         "\n"
         r"constant while $\delta_{\rm m}\propto D_+$. Back-scaling applies"
         "\n"
         rf"$D_+$ to it anyway, shrinking the split by ${1/Dplus:.0f}\times$.",
         transform=a11.transAxes, fontsize=7.5, va="top", color="0.3")

fig.suptitle(r"monofonIC $\tt LagrangianDensityOnly$: per-species Lagrangian density $\delta(q)$ at $z=200$"
             "\n"
             rf"$N={N}^3$, $L={L:g}\,$Mpc$/h$, $k_{{\rm Ny}}={kny:.3f}\,h/$Mpc, $D_+={Dplus:.4g}$, "
             r"seed 12345, $\tt DoFixing=yes$, 2LPT, CV_22 cosmology",
             fontsize=10.5, y=0.98)
fig.savefig(FPNG, dpi=DPI, bbox_inches="tight")
print("saved", FPNG)
