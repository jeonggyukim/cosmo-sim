"""Power-spectrum comparison: bin-avg delta vs bin-avg phi vs k-truncation.

For each approach, compute the binned 2D power spectrum of the lo-res
delta on the coarse grid and overlay them with the ideal lowpass
reference (= Approach B) and the input theory T^2(k) P_mu(k) = k^n_pk.

This isolates the spectral signature of each approach's failure mode:
A1 and A2 show the cell-window deficit + alias bumps; B matches the
truncated-input theory exactly.
"""
import numpy as np
import matplotlib.pyplot as plt

# Same setup as the main demo so spectra are directly comparable.
rng = np.random.default_rng(42)
N, N2, L = 64, 128, 1.0
n_pk = -2.0
dx_f = L / N2
dx_c = L / N

kx = np.fft.fftfreq(N2, d=dx_f) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
K2_f = KX**2 + KY**2
T_f  = np.where(K2_f > 0, K2_f**(0.25*n_pk), 0.0)
inv_K2_f = np.where(K2_f > 0, 1.0/K2_f, 0.0)

kc = np.fft.fftfreq(N, d=dx_c) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')
K2_c = KXc**2 + KYc**2
inv_K2_c = np.where(K2_c > 0, 1.0/K2_c, 0.0)

mu = rng.standard_normal((N2, N2))
mu_k = np.fft.fftn(mu)
delta_k_2N = T_f * mu_k; delta_k_2N[0, 0] = 0.0
delta_2N = np.fft.ifftn(delta_k_2N).real
phi_k_2N = -delta_k_2N * inv_K2_f
phi_2N   = np.fft.ifftn(phi_k_2N).real

def restrict(field):
    return field.reshape(N, 2, N, 2).mean(axis=(1, 3))

def laplacian(field, kxg, kyg):
    fk = np.fft.fftn(field)
    return np.fft.ifftn(-(kxg**2 + kyg**2) * fk).real

# Approach A1
delta_N_A1 = restrict(delta_2N)

# Approach A2 (effective density implied by bin-avg phi)
phi_N_A2 = restrict(phi_2N)
delta_eff_A2 = laplacian(phi_N_A2, KXc, KYc)

# Approach B (k-truncate mu, sample on coarse grid)
mask = (np.abs(KX) <= np.pi/dx_c) & (np.abs(KY) <= np.pi/dx_c)
mu_k_trunc = mu_k * mask
delta_k_lp = T_f * mu_k_trunc; delta_k_lp[0, 0] = 0.0
delta_lp_fine = np.fft.ifftn(delta_k_lp).real
delta_N_B = delta_lp_fine[::2, ::2]

# Per-mode amplitudes (normalize by per-grid total samples, "per-mode amplitude" convention)
def pma_pk(field, ngrid):
    fk = np.fft.fftn(field) / ngrid**2
    return np.abs(fk)**2

P2N    = pma_pk(delta_2N, N2)
P_A1   = pma_pk(delta_N_A1,    N)
P_A2   = pma_pk(delta_eff_A2,  N)
P_B    = pma_pk(delta_N_B,     N)

# Radial binning.
def radial_mean(P, K, nbins=18, kmax=None):
    K = K.ravel(); P = P.ravel()
    m = K > 0; K, P = K[m], P[m]
    if kmax is None: kmax = K.max()
    bins = np.linspace(0, kmax, nbins+1)
    idx = np.digitize(K, bins) - 1
    Pbar, kbar, cnt = [np.zeros(nbins) for _ in range(3)]
    for i in range(nbins):
        sel = idx == i
        if sel.any():
            Pbar[i] = P[sel].mean()
            kbar[i] = K[sel].mean()
            cnt[i]  = sel.sum()
    keep = cnt > 0
    return kbar[keep], Pbar[keep]

K_2N = np.sqrt(KX**2 + KY**2)
K_C  = np.sqrt(KXc**2 + KYc**2)
k2N_bar, P2N_bar = radial_mean(P2N, K_2N)
kC_bar,  PA1_bar = radial_mean(P_A1, K_C)
_,       PA2_bar = radial_mean(P_A2, K_C)
_,       PB_bar  = radial_mean(P_B,  K_C)

# Theory: P(k) propto k^n. Normalize so the smallest fine bin matches the data.
# (Same convention as plot_restriction_density.py.)
P_theory_2N = (k2N_bar**n_pk) / N2**2
P_theory_C  = (kC_bar**n_pk) / N2**2     # ideal coarse spectrum (B should match)

# Figure: two panels — P(k) overlay, and ratio to ideal.
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: power spectra
ax[0].loglog(k2N_bar, P2N_bar, 'o',  color='gray', alpha=0.6, ms=4,
             label=r'$P_{2N}(k)$ measured (fine)')
ax[0].loglog(k2N_bar, P_theory_2N, '-', color='black', lw=1,
             label=r'theory $\propto k^{n}$')
ax[0].loglog(kC_bar, PA1_bar, 's-',  color='C1', ms=4,
             label=r'A1 (bin-avg $\delta$)')
ax[0].loglog(kC_bar, PA2_bar, 'd-',  color='C2', ms=4,
             label=r'A2 (bin-avg $\phi^{(1)}$)')
ax[0].loglog(kC_bar, PB_bar,  'o-',  color='C0', ms=4,
             label=r'B (k-trunc $\mu$)')
ax[0].axvline(np.pi/dx_c, color='C1', ls=':', lw=0.8, label=r'coarse Nyquist')
ax[0].axvline(np.pi/dx_f, color='C2', ls=':', lw=0.8, label=r'fine Nyquist')
ax[0].set_xlabel(r'$|k|$'); ax[0].set_ylabel(r'$P(k)$')
ax[0].set_title('binned power spectra')
ax[0].legend(fontsize=8)

# Panel 2: ratio to ideal coarse theory
ax[1].plot(kC_bar, PA1_bar / P_theory_C, 's-', color='C1', label=r'A1 / theory')
ax[1].plot(kC_bar, PA2_bar / P_theory_C, 'd-', color='C2', label=r'A2 / theory')
ax[1].plot(kC_bar, PB_bar  / P_theory_C, 'o-', color='C0', label=r'B / theory')
ax[1].axhline(1, color='gray', lw=0.5)
ax[1].axvline(np.pi/dx_c, color='C1', ls=':', lw=0.8, label=r'coarse Nyquist')
ax[1].set_xlabel(r'$|k|$'); ax[1].set_ylabel(r'$P_{\rm lo}(k) / P_{\rm theory}(k)$')
ax[1].set_xscale('log')
ax[1].set_ylim(0, 2.0)
ax[1].set_title('ratio to ideal coarse theory')
ax[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('zoom_ic_pk.pdf')
print('wrote zoom_ic_pk.pdf')
