"""Figure: 2D restriction of delta — real-space, Fourier maps, mode-ratio vs sinc^2."""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
N, N2, L = 64, 128, 1.0
n_pk = -2.0

kx = np.fft.fftfreq(N2, d=L/N2) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
Kmag = np.sqrt(KX**2 + KY**2)
Pk   = np.where(Kmag > 0, Kmag**n_pk, 0.0)

white   = rng.standard_normal((N2, N2))
delta_k = np.fft.fftn(white) * np.sqrt(Pk); delta_k[0,0] = 0.0
delta_2N = np.fft.ifftn(delta_k).real
delta_2N -= delta_2N.mean()
delta_N  = delta_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))

dk_2N = np.fft.fftn(delta_2N) / N2**2
dk_N  = np.fft.fftn(delta_N)  / N**2
kc = np.fft.fftfreq(N, d=L/N) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')
Kc = np.sqrt(KXc**2 + KYc**2)
nc = np.fft.fftfreq(N, d=1/N).astype(int)
ix = np.where(nc >= 0, nc, nc + N2)
dk_2N_at_coarse = dk_2N[np.ix_(ix, ix)]
ratio = np.abs(dk_N) / (np.abs(dk_2N_at_coarse) + 1e-30)

dx_f = L / N2

fig, axarr = plt.subplots(2, 3, figsize=(15, 9))
ax = axarr.flatten()

vmax = max(abs(delta_2N).max(), abs(delta_N).max())
real_ext = [0, L, 0, L]
im = ax[0].imshow(delta_2N, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  origin='lower', extent=real_ext)
ax[0].set_title(r'$\delta_{2N}$ (fine)')
ax[0].set_xlabel('x'); ax[0].set_ylabel('y')
plt.colorbar(im, ax=ax[0], fraction=0.045)
im = ax[1].imshow(delta_N, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  origin='lower', extent=real_ext)
ax[1].set_title(r'$\delta_N$ (restricted)')
ax[1].set_xlabel('x'); ax[1].set_ylabel('y')
plt.colorbar(im, ax=ax[1], fraction=0.045)

def shifted_log(d):
    return np.log10(np.abs(np.fft.fftshift(d)) + 1e-12)
kfund = 2*np.pi / L
kNy_f = np.pi / dx_f
fourier_ext = [-kNy_f, kNy_f, -kNy_f, kNy_f]
im = ax[2].imshow(shifted_log(dk_2N), cmap='viridis', origin='lower',
                  extent=fourier_ext)
ax[2].set_title(r'$\log_{10}|\hat\delta_{2N}(k)|$')
ax[2].set_xlabel(r'$k_x$'); ax[2].set_ylabel(r'$k_y$')
plt.colorbar(im, ax=ax[2], fraction=0.045)

Kflat = Kc.ravel(); Rflat = ratio.ravel()
W2_2d = (np.cos(KXc*dx_f/2) * np.cos(KYc*dx_f/2))**2  # exact 2D W^2
W2flat = W2_2d.ravel()
m = Kflat > 0
ax[3].semilogy(Kflat[m], Rflat[m], '.', ms=2, alpha=0.3,
               color='C0', label='measured')
ax[3].semilogy(Kflat[m], W2flat[m], '.', ms=2, alpha=0.3,
               color='green', label=r'$W^2(k_x,k_y)$ (no aliases)')
ks = np.linspace(0, Kc.max(), 400)
ax[3].semilogy(ks, np.cos(ks*dx_f/2)**2, 'r-',
               label=r'$\cos^2(k\Delta x_f/2)$ (1D proxy)')
ax[3].set_xlabel(r'$|k|$')
ax[3].set_ylabel(r'$|\hat\delta_N|/|\hat\delta_{2N}|$')
ax[3].set_ylim(1e-3, 3); ax[3].legend(fontsize=8)

# Panel 5: binned P_N / P_2N ratio with theory prediction.
# Need P_2N at all fine modes and P_N at coarse modes (per-mode |dk|^2).
P_2N = (np.abs(dk_2N)**2)
P_N  = (np.abs(dk_N )**2)
# Theory prediction: alias-sum / P_2N_at_coarse, computed mode by mode.
# theory_PN(k_c) = sum_n W^2(k_c + pi n / dx_f) * P_2N(k_c + pi n / dx_f)
# Loop over n in {0,1}^2.
theory_PN = np.zeros_like(P_N)
for nx in (0, 1):
    for ny in (0, 1):
        # Shift each fine-grid index axis by N*nx (= pi/dx_f in mode units).
        ix_shift = (ix + N*nx) % N2
        iy_shift = (ix + N*ny) % N2
        KX_at = kx[ix_shift][:, None]            # k_x at the shifted modes
        KY_at = kx[iy_shift][None, :]
        W2_at = (np.cos(KX_at*dx_f/2) * np.cos(KY_at*dx_f/2))**2
        P2N_at = P_2N[np.ix_(ix_shift, iy_shift)]
        theory_PN += W2_at * P2N_at

# Bin both PN and theory_PN by |k_c|.
def radial_mean(field, K, nbins=20, kmax=None):
    K = K.ravel(); F = field.ravel()
    m = K > 0
    K = K[m]; F = F[m]
    if kmax is None: kmax = K.max()
    bins = np.linspace(0, kmax, nbins+1)
    idx = np.digitize(K, bins) - 1
    Fbar = np.zeros(nbins); kbar = np.zeros(nbins); cnt = np.zeros(nbins)
    for i in range(nbins):
        sel = idx == i
        if sel.any():
            Fbar[i] = F[sel].mean()
            kbar[i] = K[sel].mean()
            cnt[i]  = sel.sum()
    keep = cnt > 0
    return kbar[keep], Fbar[keep]

kbar, PN_bar     = radial_mean(P_N,      Kc)
_,    P2N_at_bar = radial_mean(P_2N[np.ix_(ix, ix)], Kc)
_,    th_bar     = radial_mean(theory_PN, Kc)

ax[4].plot(kbar, PN_bar/P2N_at_bar, 'o-', color='C0',
           label=r'measured $P_N/P_{2N}$ (binned)')
ax[4].plot(kbar, th_bar/P2N_at_bar, 's--', color='red',
           label=r'theory: $\sum_{\bf n}|W|^2 P_{2N}/P_{2N}$')
ax[4].axhline(1, color='gray', lw=0.5)
ax[4].set_xlabel(r'$|k_c|$')
ax[4].set_ylabel(r'$P_N(k_c) / P_{2N}(k_c)$')
ax[4].set_title('binned power ratio (alias-sum check)')
ax[4].legend(fontsize=8)

# Panel 6: binned P_N(k), P_2N(k), with theory predictions overlaid.
kbar_2N, P2N_bar = radial_mean(P_2N, np.sqrt(KX**2+KY**2))
# Theory P_2N from the analytical normalization: with the FFT convention
# dk = fftn(delta) / N2**2, the per-mode expectation is <|dk|^2> = P(k)/N2**2.
P_input = (kbar_2N**n_pk) / N2**2
# Theory P_N: alias-sum prediction, already binned as `th_bar`.
ax[5].loglog(kbar_2N, P2N_bar, 'o', color='gray', alpha=0.7,
             label=r'$P_{2N}(k)$ measured')
ax[5].loglog(kbar_2N, P_input, '-', color='black', lw=1,
             label=r'theory $\propto k^{n}$')
ax[5].loglog(kbar,    PN_bar,  's', color='C0', alpha=0.7,
             label=r'$P_N(k_c)$ measured')
ax[5].loglog(kbar,    th_bar,  '--', color='red',
             label=r'$P_N$ theory (alias-sum)')
ax[5].axvline(np.pi/dx_f/2, color='C1', ls=':', lw=0.8,
              label=r'coarse Nyquist')
ax[5].axvline(np.pi/dx_f,   color='C2', ls=':', lw=0.8,
              label=r'fine Nyquist')
ax[5].set_xlabel(r'$|k|$')
ax[5].set_ylabel(r'$P(k)$')
ax[5].set_title('binned power spectra')
ax[5].legend(fontsize=7)

plt.tight_layout()
plt.savefig('restriction_density.pdf')
print("wrote restriction_density.pdf")
