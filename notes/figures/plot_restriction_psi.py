"""Figure: 1LPT displacement Psi^(1) under restriction — fine-then-restrict vs restrict-then-solve."""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
N, N2, L = 64, 128, 1.0
n_pk = -2.0

kx = np.fft.fftfreq(N2, d=L/N2) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
K2 = KX**2 + KY**2
Pk = np.where(K2 > 0, K2**(0.5*n_pk), 0.0)

white = rng.standard_normal((N2, N2))
dk = np.fft.fftn(white) * np.sqrt(Pk); dk[0,0] = 0.0
delta_2N = np.fft.ifftn(dk).real
delta_2N -= delta_2N.mean()
delta_N  = delta_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))

def psi1(delta, kx_g, ky_g):
    K2 = kx_g**2 + ky_g**2
    inv = np.where(K2 > 0, 1.0/K2, 0.0)
    dh  = np.fft.fftn(delta)
    return (np.fft.ifftn(1j*kx_g*inv*dh).real,
            np.fft.ifftn(1j*ky_g*inv*dh).real)

psix_2N, psiy_2N = psi1(delta_2N, KX, KY)
psix_2N_avg = psix_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))
psiy_2N_avg = psiy_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))

kc = np.fft.fftfreq(N, d=L/N) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')
psix_N, psiy_N = psi1(delta_N, KXc, KYc)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))

vmax = max(abs(psix_2N_avg).max(), abs(psix_N).max())
im = ax[0].imshow(psix_2N_avg, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0].set_title(r'$\overline{\Psi_x^{(1)}[\delta_{2N}]}$ (fine, then restrict)')
plt.colorbar(im, ax=ax[0], fraction=0.045)

im = ax[1].imshow(psix_N, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[1].set_title(r'$\Psi_x^{(1)}[\delta_N]$ (restrict, then solve)')
plt.colorbar(im, ax=ax[1], fraction=0.045)

diff = psix_2N_avg - psix_N
dmax = max(abs(diff).max(), 1e-30)
im = ax[2].imshow(diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax, origin='lower')
ax[2].set_title('difference')
plt.colorbar(im, ax=ax[2], fraction=0.045)

plt.tight_layout()
plt.savefig('restriction_psi.pdf')
print("wrote restriction_psi.pdf")
