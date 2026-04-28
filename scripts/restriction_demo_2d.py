"""
restriction_demo_2d.py — 2D version of restriction_demo.py.

Build a 2D Gaussian random field on a 2N x 2N grid with P(k) propto k^n,
restrict by 2x2 block averaging to N x N, and compare:
  (1) real-space fields and their x-gradients,
  (2) 2D |delta(k)| maps,
  (3) azimuthally-binned P(k) and the mode-by-mode amplitude ratio vs sinc^2.

Restriction = convolution with top-hat of width dx_fine, then 2x subsample.
In Fourier space:
    delta_N(k) = sum_n W(k + 2 pi n / dx_coarse) delta_{2N}(k + 2 pi n / dx_coarse)
with W(k) = sinc(k_x dx_f/2) sinc(k_y dx_f/2).
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
N    = 128
N2   = 2 * N
L    = 1.0
n_pk = -1.0

# ---- fine field with P(k) propto k^n ----
kx = np.fft.fftfreq(N2, d=L/N2) * 2 * np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
Kmag = np.sqrt(KX**2 + KY**2)
Pk   = np.where(Kmag > 0, Kmag**n_pk, 0.0)

white  = rng.standard_normal((N2, N2))
delta_k = np.fft.fftn(white) * np.sqrt(Pk)
delta_k[0, 0] = 0.0
delta_2N = np.fft.ifftn(delta_k).real
delta_2N -= delta_2N.mean()

# ---- restrict: average 2x2 blocks ----
delta_N = delta_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))

# ---- FFTs ----
dk_2N = np.fft.fftn(delta_2N) / N2**2
dk_N  = np.fft.fftn(delta_N)  / N**2

kc = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')
Kc = np.sqrt(KXc**2 + KYc**2)

nc = np.fft.fftfreq(N, d=1/N).astype(int)
ix = np.where(nc >= 0, nc, nc + N2)
dk_2N_at_coarse = dk_2N[np.ix_(ix, ix)]

dx_f = L / N2
W = np.sinc(KXc*dx_f/(2*np.pi)) * np.sinc(KYc*dx_f/(2*np.pi))

# ---- spectral d_x delta ----
gx_2N = np.fft.ifftn(1j * KX  * np.fft.fftn(delta_2N)).real
gx_N  = np.fft.ifftn(1j * KXc * np.fft.fftn(delta_N )).real
gx_2N_avg = gx_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))

# ---- binned P(k) ----
def binned_pk(dk, K, L, nbin=30):
    P = (np.abs(dk)**2) * L**2
    bins = np.linspace(0, K.max(), nbin+1)
    idx  = np.digitize(K.ravel(), bins) - 1
    Pbin = np.zeros(nbin); kbin = np.zeros(nbin); cnt = np.zeros(nbin)
    for i in range(nbin):
        m = idx == i
        if m.any():
            Pbin[i] = P.ravel()[m].mean()
            kbin[i] = K.ravel()[m].mean()
            cnt[i]  = m.sum()
    return kbin[cnt>0], Pbin[cnt>0]

k_f, P_f = binned_pk(dk_2N, np.sqrt(KX**2+KY**2), L)
k_c, P_c = binned_pk(dk_N,  Kc, L)

ratio = np.abs(dk_N) / (np.abs(dk_2N_at_coarse) + 1e-30)
print(f"<|delta_N(k)| / |delta_2N(k)|>  = {ratio.mean():.4f}")
print(f"<W(k) at coarse modes>          = {W.mean():.4f}")
print(f"std(delta_2N) = {delta_2N.std():.4e}, std(delta_N) = {delta_N.std():.4e}")

# ---- plots ----
fig, ax = plt.subplots(2, 3, figsize=(13, 8))

vmax = max(abs(delta_2N).max(), abs(delta_N).max())
im = ax[0,0].imshow(delta_2N, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,0].set_title(r'$\delta_{2N}$ (fine)'); plt.colorbar(im, ax=ax[0,0], fraction=0.045)
im = ax[0,1].imshow(delta_N,  cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,1].set_title(r'$\delta_N$ (restricted)'); plt.colorbar(im, ax=ax[0,1], fraction=0.045)

# difference: block-avg(fine) - restricted = 0 by construction, so instead
# show block-avg of fine gradient vs gradient of restricted field.
gmax = max(abs(gx_2N_avg).max(), abs(gx_N).max())
im = ax[0,2].imshow(gx_2N_avg - gx_N, cmap='RdBu_r',
                    vmin=-gmax/4, vmax=gmax/4, origin='lower')
ax[0,2].set_title(r'$\overline{\partial_x\delta_{2N}} - \partial_x\delta_N$')
plt.colorbar(im, ax=ax[0,2], fraction=0.045)

# 2D |delta(k)| maps (fftshifted), log scale
def shifted_log(d):
    return np.log10(np.abs(np.fft.fftshift(d)) + 1e-12)
im = ax[1,0].imshow(shifted_log(dk_2N), cmap='viridis', origin='lower')
ax[1,0].set_title(r'$\log_{10}|\delta_{2N}(k)|$'); plt.colorbar(im, ax=ax[1,0], fraction=0.045)
im = ax[1,1].imshow(shifted_log(dk_N), cmap='viridis', origin='lower')
ax[1,1].set_title(r'$\log_{10}|\delta_N(k)|$'); plt.colorbar(im, ax=ax[1,1], fraction=0.045)

# P(k) and ratio
ax[1,2].loglog(k_f, P_f, label=r'$P(k)$ on $2N$')
ax[1,2].loglog(k_c, P_c, '--', label=r'$P(k)$ on $N$')
ax[1,2].axvline(np.pi*N/L,  color='k', ls=':', label=r'$k_{Ny,N}$')
ax[1,2].axvline(np.pi*N2/L, color='gray', ls=':', label=r'$k_{Ny,2N}$')
ax[1,2].set_xlabel('k'); ax[1,2].set_ylabel('P(k)'); ax[1,2].legend()

plt.tight_layout()
plt.savefig('restriction_demo_2d.png', dpi=120)
print("wrote restriction_demo_2d.png")

# ---- 1D slices: rho(x, y=fixed) and |rho_hat(kx, ky=0)| ----
fig_s, axs = plt.subplots(1, 2, figsize=(12, 4))

# real-space slice along x at the central row
x_f = (np.arange(N2) + 0.5) * (L / N2)
x_c = (np.arange(N)  + 0.5) * (L / N)
axs[0].plot(x_f, delta_2N[:, N2//2], label=r'$\delta_{2N}(x, y_0)$', lw=1)
axs[0].plot(x_c, delta_N [:, N //2], 'o-', ms=3,
            label=r'$\delta_N(x, y_0)$ (restricted)')
axs[0].set_xlabel('x'); axs[0].set_ylabel(r'$\delta$')
axs[0].set_title('real-space slice'); axs[0].legend()

# Fourier-space slice along kx at ky = 0
kx_f_sorted = np.fft.fftshift(kx)
kx_c_sorted = np.fft.fftshift(kc)
amp_f_slice = np.abs(np.fft.fftshift(dk_2N, axes=(0,1)))[:, N2//2]
amp_c_slice = np.abs(np.fft.fftshift(dk_N,  axes=(0,1)))[:, N //2]
axs[1].semilogy(kx_f_sorted, amp_f_slice, label=r'$|\hat\delta_{2N}(k_x, 0)|$', lw=1)
axs[1].semilogy(kx_c_sorted, amp_c_slice, 'o-', ms=3,
                label=r'$|\hat\delta_N(k_x, 0)|$')
axs[1].axvline( np.pi*N/L, color='k', ls=':', label=r'$\pm k_{Ny,N}$')
axs[1].axvline(-np.pi*N/L, color='k', ls=':')
axs[1].set_xlabel(r'$k_x$'); axs[1].set_ylabel(r'$|\hat\delta|$')
axs[1].set_title(r'Fourier-space slice ($k_y=0$)'); axs[1].legend()

plt.tight_layout()
plt.savefig('restriction_demo_2d_slices.png', dpi=120)
print("wrote restriction_demo_2d_slices.png")

# extra figure: mode ratio scatter vs sinc^2 prediction
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(Kc.ravel(), ratio.ravel(), '.', ms=2, alpha=0.3, label='measured')
ks = np.linspace(0, Kc.max(), 300)
ax2.plot(ks, np.sinc(ks*dx_f/(2*np.pi))**2, 'r-',
         label=r'$\mathrm{sinc}^2(k\Delta x_f/2)$ (1D proxy)')
ax2.set_xlabel('|k|'); ax2.set_ylabel(r'$|\delta_N|/|\delta_{2N}|$')
ax2.legend(); plt.tight_layout()
plt.savefig('restriction_demo_2d_ratio.png', dpi=120)
print("wrote restriction_demo_2d_ratio.png")
