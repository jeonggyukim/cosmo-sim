"""Combined P(k) figure for the zoom-IC discussion.

Top row (restriction methods, A1/A2/B from plot_zoom_ic_comparison.py):
  - left:  P_lo(k) spectra overlay vs input theory
  - right: ratio P_lo(k) / P_theory(k)

Bottom row (2LPT + CIC particle pipeline):
  - left:  P spectra of grid input, CIC raw, CIC deconvolved vs theory
  - right: ratio to theory

Together: the top row shows that bin-averaging methods give wrong
lo-res power; the bottom row shows that the standard particle
pipeline (2LPT displacement + CIC deposit + deconvolution) recovers
the theory faithfully.
"""
import numpy as np
import matplotlib.pyplot as plt

# ===================================================================
# Common setup: same realization as plot_zoom_ic_comparison.py.
# ===================================================================
rng_seed = 42
N, N2, L = 64, 128, 1.0
n_pk = -2.0
dx_f = L / N2
dx_c = L / N

# When True, rescale each Fourier mode of the white noise to have unit
# amplitude per mode (Angulo & Pontzen 2016 "fixed-amplitude" trick).
# Phases are still random (per-realization), but |mu_k| = sqrt(N_total)
# is exactly the theory expectation for every k -- no cosmic-variance
# noise on per-mode P(k). Useful for visualising restriction systematics
# without scatter.
fix_mode_amplitude = True

# Fine k-grid.
kx = np.fft.fftfreq(N2, d=dx_f) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
K2_f = KX**2 + KY**2
T_f = np.where(K2_f > 0, K2_f**(0.25*n_pk), 0.0)
inv_K2_f = np.where(K2_f > 0, 1.0/K2_f, 0.0)

# Coarse k-grid.
kc = np.fft.fftfreq(N, d=dx_c) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')
K2_c = KXc**2 + KYc**2
inv_K2_c = np.where(K2_c > 0, 1.0/K2_c, 0.0)

# Generate fine-grid delta from white noise.
rng = np.random.default_rng(rng_seed)
mu_2N = rng.standard_normal((N2, N2))
mu_k = np.fft.fftn(mu_2N)
if fix_mode_amplitude:
    # Set |mu_k| = sqrt(N_total) per mode (theory expectation for unit-variance
    # white noise), keeping the random phase. Hermitian symmetry preserved
    # because conjugate-pair magnitudes are equal and we rescale each by the
    # same factor.
    target = np.sqrt(N2*N2)
    amp = np.abs(mu_k)
    mu_k = np.where(amp > 0, mu_k / amp * target, mu_k)
delta_k_2N = T_f * mu_k; delta_k_2N[0, 0] = 0.0
delta_2N = np.fft.ifftn(delta_k_2N).real
phi_k_2N = -delta_k_2N * inv_K2_f
phi_2N = np.fft.ifftn(phi_k_2N).real

def restrict(field):
    return field.reshape(N, 2, N, 2).mean(axis=(1, 3))

def laplacian(field, kxg, kyg):
    fk = np.fft.fftn(field)
    return np.fft.ifftn(-(kxg**2 + kyg**2) * fk).real

def pma_pk(field, ngrid):
    fk = np.fft.fftn(field) / ngrid**2
    return np.abs(fk)**2

def radial_mean(P, K, nbins=22, kmax=None):
    K = K.ravel(); P = P.ravel()
    m = K > 0; K, P = K[m], P[m]
    if kmax is None: kmax = K.max()
    # Log-spaced bins (better for steep power-law spectra: bin centres
    # cluster more uniformly per decade and Jensen artefacts shrink).
    bins = np.logspace(np.log10(K.min()*0.999), np.log10(kmax*1.001), nbins+1)
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

# ===================================================================
# Top row: restriction methods on the coarse grid.
# ===================================================================
delta_N_A1 = restrict(delta_2N)
phi_N_A2   = restrict(phi_2N)
delta_eff_A2 = laplacian(phi_N_A2, KXc, KYc)
mask = (np.abs(KX) <= np.pi/dx_c) & (np.abs(KY) <= np.pi/dx_c)
delta_k_lp = T_f * (mu_k * mask); delta_k_lp[0, 0] = 0.0
delta_lp_fine = np.fft.ifftn(delta_k_lp).real
delta_N_B = delta_lp_fine[::2, ::2]

P_A1 = pma_pk(delta_N_A1, N)
P_A2 = pma_pk(delta_eff_A2, N)
P_B  = pma_pk(delta_N_B, N)
P_2N = pma_pk(delta_2N, N2)

K2N_mag = np.sqrt(K2_f)
KC_mag  = np.sqrt(K2_c)
k2N_bar, P2N_bar = radial_mean(P_2N, K2N_mag)
kC_bar,  PA1_bar = radial_mean(P_A1, KC_mag)
_,       PA2_bar = radial_mean(P_A2, KC_mag)
_,       PB_bar  = radial_mean(P_B,  KC_mag)
P_theory_C  = (kC_bar**n_pk)  / N2**2
P_theory_2N = (k2N_bar**n_pk) / N2**2

# Normalize x-axis by the coarse Nyquist wavenumber, so x = 1 marks
# the coarse Nyquist and x = 2 the fine Nyquist regardless of N.
k_Ny_C = np.pi / dx_c
xC  = kC_bar  / k_Ny_C
x2N = k2N_bar / k_Ny_C

# ===================================================================
# Bottom row: 2LPT + CIC particle pipeline (on the FINE grid only).
# ===================================================================
psi1_x = np.fft.ifftn(1j*KX*inv_K2_f*delta_k_2N).real
psi1_y = np.fft.ifftn(1j*KY*inv_K2_f*delta_k_2N).real
psi1_xx = np.fft.ifftn(1j*KX*np.fft.fftn(psi1_x)).real
psi1_xy = np.fft.ifftn(1j*KY*np.fft.fftn(psi1_x)).real
psi1_yy = np.fft.ifftn(1j*KY*np.fft.fftn(psi1_y)).real
S2 = psi1_xx*psi1_yy - psi1_xy**2
S2_k = np.fft.fftn(S2)
psi2_x = np.fft.ifftn(1j*KX*inv_K2_f*S2_k).real
psi2_y = np.fft.ifftn(1j*KY*inv_K2_f*S2_k).real
disp_x = psi1_x - (3.0/7.0)*psi2_x
disp_y = psi1_y - (3.0/7.0)*psi2_y

q_axis = (np.arange(N2) + 0.5) * dx_f
QX, QY = np.meshgrid(q_axis, q_axis, indexing='ij')
x_part = (QX + disp_x) % L
y_part = (QY + disp_y) % L

def cic_deposit_2d(x, y, ngrid, box_size):
    cell = box_size / ngrid
    fx = x / cell - 0.5
    fy = y / cell - 0.5
    i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int)
    dx_p = fx - i0; dy_p = fy - j0
    i1 = (i0 + 1) % ngrid; j1 = (j0 + 1) % ngrid
    i0 %= ngrid; j0 %= ngrid
    rho = np.zeros((ngrid, ngrid))
    for w, ii, jj in [
        ((1-dx_p)*(1-dy_p), i0, j0),
        (   dx_p *(1-dy_p), i1, j0),
        ((1-dx_p)*   dy_p , i0, j1),
        (   dx_p *   dy_p , i1, j1)]:
        np.add.at(rho, (ii, jj), w)
    return rho/rho.mean() - 1.0

delta_cic = cic_deposit_2d(x_part.ravel(), y_part.ravel(), N2, L)
P_cic = pma_pk(delta_cic, N2)
W_CIC = np.sinc(KX*dx_f/(2*np.pi)) * np.sinc(KY*dx_f/(2*np.pi))
P_cic_dec = P_cic / np.where(W_CIC**2 > 1e-8, W_CIC**2, 1.0)
_, Pcic_bar = radial_mean(P_cic, K2N_mag)
_, Pdec_bar = radial_mean(P_cic_dec, K2N_mag)

# ===================================================================
# Plot
# ===================================================================
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# --- Top-left: restriction-methods P(k) overlay.
ax[0,0].loglog(x2N, P2N_bar, 'o', color='gray', alpha=0.6, ms=4,
               label=r'$P_{2N}(k)$ measured (fine)')
ax[0,0].loglog(x2N, P_theory_2N, 'k-', lw=1, label=fr'theory $\propto k^{{n}}$ ($n={n_pk:g}$)')
ax[0,0].loglog(xC, PA1_bar, 's-', color='C1', ms=4,
               label=r'A1 (bin-avg $\delta$)')
ax[0,0].loglog(xC, PA2_bar, 'd-', color='C2', ms=4,
               label=r'A2 (bin-avg $\phi^{(1)}$)')
ax[0,0].loglog(xC, PB_bar,  'o-', color='C0', ms=4,
               label=r'B (k-trunc $\mu$)')
ax[0,0].axvline(1.0, color='C1', ls=':', lw=0.8, label=r'coarse Nyquist')
ax[0,0].axvline(2.0, color='C2', ls=':', lw=0.8, label=r'fine Nyquist')
ax[0,0].set_xlabel(r'$|k| / k_{\rm Ny,N}$'); ax[0,0].set_ylabel(r'$P(k)$')
ax[0,0].set_title('restriction methods: lo-res $P(k)$ on coarse grid ($N\\times N$)')
ax[0,0].legend(fontsize=8)

# --- Top-right: ratio to theory.
ax[0,1].plot(xC, PA1_bar/P_theory_C, 's-', color='C1', label='A1 / theory')
ax[0,1].plot(xC, PA2_bar/P_theory_C, 'd-', color='C2', label='A2 / theory')
ax[0,1].plot(xC, PB_bar /P_theory_C, 'o-', color='C0', label='B / theory')
ax[0,1].axhline(1, color='gray', lw=0.5)
ax[0,1].axvline(1.0, color='C1', ls=':', lw=0.8)
ax[0,1].set_xscale('log'); ax[0,1].set_ylim(0, 2)
ax[0,1].set_xlabel(r'$|k| / k_{\rm Ny,N}$'); ax[0,1].set_ylabel(r'$P_{\rm lo} / P_{\rm theory}$')
ax[0,1].set_title('restriction methods: $P_{\\rm lo}/P_{\\rm theory}$ (coarse grid)')
ax[0,1].legend(fontsize=8)

# --- Bottom-left: 2LPT+CIC P(k) overlay.
ax[1,0].loglog(x2N, P_theory_2N, 'k-', lw=1, label=fr'theory $\propto k^{{n}}$ ($n={n_pk:g}$)')
ax[1,0].loglog(x2N, P2N_bar, 'o', color='gray', alpha=0.7, ms=5,
               label=r'$P[\delta_{\rm grid}]$ (input on grid)')
ax[1,0].loglog(x2N, Pcic_bar, 's', color='C1', alpha=0.7, ms=5,
               label=r'$P[\delta_{\rm CIC}]$ (raw, after CIC)')
ax[1,0].loglog(x2N, Pdec_bar, 'd', color='C0', alpha=0.7, ms=5,
               label=r'$P[\delta_{\rm CIC}] / W_{\rm CIC}^2$ (deconvolved)')
ax[1,0].axvline(2.0, color='gray', ls=':', label=r'fine Nyquist')
ax[1,0].set_xlabel(r'$|k| / k_{\rm Ny,N}$'); ax[1,0].set_ylabel(r'$P(k)$')
ax[1,0].set_title('2LPT particles + CIC on fine grid ($2N \\times 2N$)')
ax[1,0].legend(fontsize=8)

# --- Bottom-right: ratio to theory.
ax[1,1].plot(x2N, P2N_bar /P_theory_2N, 'o-', color='gray',
             label=r'grid input / theory')
ax[1,1].plot(x2N, Pcic_bar/P_theory_2N, 's-', color='C1',
             label=r'CIC raw / theory')
ax[1,1].plot(x2N, Pdec_bar/P_theory_2N, 'd-', color='C0',
             label=r'CIC deconvolved / theory')
ax[1,1].axhline(1, color='gray', lw=0.5)
ax[1,1].axvline(2.0, color='gray', ls=':')
ax[1,1].set_xscale('log'); ax[1,1].set_ylim(0, 2)
ax[1,1].set_xlabel(r'$|k| / k_{\rm Ny,N}$'); ax[1,1].set_ylabel(r'ratio to theory')
ax[1,1].set_title('2LPT + CIC: $P/P_{\\rm theory}$ (fine grid)')
ax[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('zoom_ic_pk_full.pdf')
print('wrote zoom_ic_pk_full.pdf')
