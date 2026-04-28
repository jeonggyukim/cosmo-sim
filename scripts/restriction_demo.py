"""
restriction_demo.py — compare delta_{2N} vs restricted delta_N in Fourier space and gradient.

The 2x2x2 cell average is convolution with a real-space top-hat of width dx_fine,
followed by 2x subsampling onto the coarse grid. In Fourier space:
    delta_N(k) = sum_n W(k + 2 pi n / dx_coarse) delta_{2N}(k + 2 pi n / dx_coarse)
where W(k) = prod_a sinc(k_a dx_fine / 2) and the sum over n folds in aliases
from |k_a| > k_Ny,coarse = pi / dx_coarse.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
N    = 64           # coarse grid
N2   = 2 * N        # fine grid
L    = 1.0          # box size (arbitrary units)
n_pk = -1.0         # P(k) propto k^n_pk for the synthetic field

# ---- build a Gaussian random field on the fine grid with P(k) propto k^n ----
kx = np.fft.fftfreq(N2, d=L/N2) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
Pk   = np.where(Kmag > 0, Kmag**n_pk, 0.0)

white = rng.standard_normal((N2, N2, N2))
delta_k = np.fft.fftn(white) * np.sqrt(Pk)
delta_k[0, 0, 0] = 0.0
delta_2N = np.fft.ifftn(delta_k).real
delta_2N -= delta_2N.mean()

# ---- restrict: average 2x2x2 blocks (mass-conserving in the cell-average sense) ----
delta_N = delta_2N.reshape(N, 2, N, 2, N, 2).mean(axis=(1, 3, 5))

# ---- FFTs ----
dk_2N = np.fft.fftn(delta_2N) / N2**3
dk_N  = np.fft.fftn(delta_N)  / N**3

kc = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
KXc, KYc, KZc = np.meshgrid(kc, kc, kc, indexing='ij')
Kc = np.sqrt(KXc**2 + KYc**2 + KZc**2)

# Index the coarse modes inside the fine cube (low-frequency block)
nc = np.fft.fftfreq(N, d=1/N).astype(int)
ix = np.where(nc >= 0, nc, nc + N2)
dk_2N_at_coarse = dk_2N[np.ix_(ix, ix, ix)]

# Top-hat (cell-average) window on the fine grid spacing dx_f = L/N2
dx_f = L / N2
W = (np.sinc(KXc * dx_f / (2*np.pi))
   * np.sinc(KYc * dx_f / (2*np.pi))
   * np.sinc(KZc * dx_f / (2*np.pi)))

# ---- binned P(k) ----
def binned_pk(dk, K, L, nbin=20):
    P = (np.abs(dk)**2) * L**3
    kmax = K.max()
    bins = np.linspace(0, kmax, nbin+1)
    idx  = np.digitize(K.ravel(), bins) - 1
    Pbin = np.zeros(nbin); kbin = np.zeros(nbin); cnt = np.zeros(nbin)
    for i in range(nbin):
        m = idx == i
        if m.any():
            Pbin[i] = P.ravel()[m].mean()
            kbin[i] = K.ravel()[m].mean()
            cnt[i]  = m.sum()
    return kbin[cnt>0], Pbin[cnt>0]

k_f, P_f = binned_pk(dk_2N, np.sqrt(KX**2+KY**2+KZ**2), L)
k_c, P_c = binned_pk(dk_N,  Kc, L)

# ---- spectral gradient d_x delta ----
gx_2N = np.fft.ifftn(1j * KX  * np.fft.fftn(delta_2N)).real
gx_N  = np.fft.ifftn(1j * KXc * np.fft.fftn(delta_N )).real
gx_2N_avg = gx_2N.reshape(N,2,N,2,N,2).mean(axis=(1,3,5))

# ---- report ----
ratio = np.abs(dk_N) / (np.abs(dk_2N_at_coarse) + 1e-30)
print(f"<|delta_N(k)| / |delta_2N(k)|>  = {ratio.mean():.4f}")
print(f"<W(k) at coarse modes>          = {W.mean():.4f}   (expected attenuation w/o aliasing)")
print(f"std(delta_2N) = {delta_2N.std():.4e}, std(delta_N) = {delta_N.std():.4e}")
print(f"std(dx delta_2N block-avg) = {gx_2N_avg.std():.4e}, "
      f"std(dx delta_N from restricted) = {gx_N.std():.4e}")

# ---- plots ----
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

ax[0].loglog(k_f, P_f, label=r'$P(k)$ on $2N$')
ax[0].loglog(k_c, P_c, '--', label=r'$P(k)$ on $N$ (restricted)')
ax[0].axvline(np.pi*N/L,  color='k', ls=':', label=r'$k_{Ny,N}$')
ax[0].axvline(np.pi*N2/L, color='gray', ls=':', label=r'$k_{Ny,2N}$')
ax[0].set_xlabel('k'); ax[0].set_ylabel('P(k)'); ax[0].legend()

ax[1].plot(Kc.ravel(), ratio.ravel(), '.', ms=2, alpha=0.3, label='measured ratio')
ks = np.linspace(0, Kc.max(), 200)
ax[1].plot(ks, np.sinc(ks*dx_f/(2*np.pi))**3, 'r-',
           label=r'$\mathrm{sinc}^3(k\Delta x_f/2)$ (1D proxy)')
ax[1].set_xlabel('|k|'); ax[1].set_ylabel(r'$|\delta_N|/|\delta_{2N}|$'); ax[1].legend()

ax[2].plot(gx_2N_avg[:, N//2, N//2], label=r'block-avg $\partial_x\delta_{2N}$')
ax[2].plot(gx_N    [:, N//2, N//2], '--', label=r'$\partial_x\delta_N$ (after restriction)')
ax[2].set_xlabel('i'); ax[2].set_ylabel(r'$\partial_x \delta$'); ax[2].legend()

plt.tight_layout()
plt.savefig('restriction_demo.png', dpi=120)
print("wrote restriction_demo.png")
