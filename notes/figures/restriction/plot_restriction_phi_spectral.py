"""Spectral diagnostics for path-3 (restrict-phi-then-gradient) Poisson
inconsistency. Same setup as plot_restriction_phi.py.

Two panels:
  - radial power-spectrum ratio |delta_eff(k) - delta_N(k)|^2 / |delta_N(k)|^2
    vs |k|/k_Ny,N: shows that the inconsistency is concentrated near the
    coarse Nyquist.
  - histogram of (delta_eff - delta_N) / RMS(delta_N) over coarse cells,
    with RMS and max printed.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
N, N2, L = 64, 128, 1.0
n_pk = -2.0

kx = np.fft.fftfreq(N2, d=L/N2) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
K2_f = KX**2 + KY**2
Pk = np.where(K2_f > 0, K2_f**(0.5*n_pk), 0.0)

white = rng.standard_normal((N2, N2))
dk = np.fft.fftn(white) * np.sqrt(Pk); dk[0,0] = 0.0
delta_2N = np.fft.ifftn(dk).real
delta_2N -= delta_2N.mean()

kc = np.fft.fftfreq(N, d=L/N) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')
K2_c = KXc**2 + KYc**2

inv_K2_f = np.where(K2_f > 0, 1.0/K2_f, 0.0)
inv_K2_c = np.where(K2_c > 0, 1.0/K2_c, 0.0)

def restrict(f):
    return f.reshape(N, 2, N, 2).mean(axis=(1, 3))

def lap(f, kxg, kyg):
    return np.fft.ifftn(-(kxg**2 + kyg**2) * np.fft.fftn(f)).real

# Path-3 inputs.
phi_2N = np.fft.ifftn(-np.fft.fftn(delta_2N) * inv_K2_f).real
phi_N_rp = restrict(phi_2N)
delta_eff = lap(phi_N_rp, KXc, KYc)
delta_N = restrict(delta_2N)
resid = delta_eff - delta_N

# --- diagnostic 1: radial power ratio ---
def coarse_pk_radial(field, nbins=18):
    fk = np.fft.fftn(field) / N**2
    P = np.abs(fk)**2
    Kmag = np.sqrt(KXc**2 + KYc**2)
    mask = Kmag > 0
    kflat = Kmag[mask]; Pflat = P[mask]
    # Stop at axis-aligned coarse Nyquist (avoid the corner-only geometric
    # band, same convention as the analytic curve in fig 3).
    kny = np.pi * N / L
    kedges = np.linspace(0, kny, nbins+1)
    kctr = 0.5*(kedges[:-1] + kedges[1:])
    Pbin = np.zeros(nbins)
    for i in range(nbins):
        m = (kflat >= kedges[i]) & (kflat < kedges[i+1])
        Pbin[i] = Pflat[m].mean() if m.any() else np.nan
    return kctr, Pbin, kny

k_r, P_resid, kny = coarse_pk_radial(resid)
_, P_dN, _ = coarse_pk_radial(delta_N)
ratio = P_resid / P_dN

# --- diagnostic 2: histogram ---
rms_dN = np.sqrt((delta_N**2).mean())
relerr = resid.ravel() / rms_dN
rms_rel = np.sqrt((relerr**2).mean())
maxabs = np.max(np.abs(relerr))

# --- figure ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

ax[0].semilogy(k_r/kny, ratio, 'o-', ms=4)
ax[0].axhline(1.0, color='k', lw=0.8, ls=':')
ax[0].set_xlabel(r'$|k_c| / k_{\mathrm{Ny},N}$')
ax[0].set_ylabel(r'$P_{\delta_{\rm eff}-\delta_N}(k_c)\,/\,P_{\delta_N}(k_c)$')
ax[0].set_title('power-spectrum ratio of\n'
                r'$\delta_{\rm eff}-\delta_N$ to $\delta_N$')

ax[1].hist(relerr, bins=60, color='C0', edgecolor='k', linewidth=0.3)
ax[1].set_xlabel(r'$(\delta_{\rm eff} - \delta_N)\,/\,\mathrm{RMS}(\delta_N)$')
ax[1].set_ylabel('number of coarse-grid cells')
ax[1].set_title(f'per-cell relative inconsistency\n'
                f'(RMS = {rms_rel:.1%}, max = {maxabs:.1%})')

fig.suptitle(rf'Path-3 Poisson inconsistency, $P(k)\propto k^{{n}}$ with $n={n_pk:g}$')
plt.tight_layout()
plt.savefig('restriction_phi_spectral.pdf')
print('wrote restriction_phi_spectral.pdf')
print(f'rms_rel = {rms_rel:.4%}, max = {maxabs:.4%}')
print(f'P_resid/P_dN at k_Ny,N: {ratio[-1]:.3f}')
