"""
restriction_psi1_2d.py — how does restriction affect the 1LPT displacement Psi^(1)?

Linear LPT (cosmo_ic.tex Eqs. 64-67):
    nabla . Psi^(1) = -delta_lin     =>     Psi^(1)(k) = i k / k^2 * delta_lin(k)
    v^(1) = a H f Psi^(1)

We build a 2D linear delta on a 2N grid, then compare two coarse-grid Psi^(1):
    (A) fine-grid Psi from delta_{2N}, then 2x2 block-averaged           [Psi_2N_avg]
    (B) Psi solved on coarse grid from the restricted delta_N            [Psi_N]
The residual (A) - (B) is the LPT-relevant restriction error.

The kernel i k / k^2 weights LARGE scales, so Psi should be far less sensitive to
restriction than delta or grad(delta).
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
N    = 128
N2   = 2 * N
L    = 1.0
n_pk = -1.0   # try 0 or +1 to make small-scale aliasing more visible

# ---- fine field, P(k) propto k^n ----
kx = np.fft.fftfreq(N2, d=L/N2) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
K2 = KX**2 + KY**2
Pk = np.where(K2 > 0, K2**(0.5*n_pk), 0.0)

white = rng.standard_normal((N2, N2))
dk    = np.fft.fftn(white) * np.sqrt(Pk)
dk[0, 0] = 0.0
delta_2N = np.fft.ifftn(dk).real
delta_2N -= delta_2N.mean()

# ---- restrict ----
delta_N = delta_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))

# ---- Psi^(1) on the fine grid:  Psi(k) = i k / k^2 * delta(k) ----
def psi1(delta, kx_grid, ky_grid):
    K2 = kx_grid**2 + ky_grid**2
    inv = np.where(K2 > 0, 1.0/K2, 0.0)
    dhat = np.fft.fftn(delta)
    psix = np.fft.ifftn(1j * kx_grid * inv * dhat).real
    psiy = np.fft.ifftn(1j * ky_grid * inv * dhat).real
    return psix, psiy

psix_2N, psiy_2N = psi1(delta_2N, KX, KY)

# block-average each component down to coarse grid
psix_2N_avg = psix_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))
psiy_2N_avg = psiy_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))

# coarse-grid Psi from delta_N
kc = np.fft.fftfreq(N, d=L/N) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')
psix_N, psiy_N = psi1(delta_N, KXc, KYc)

# ---- summary stats ----
def relerr(a, b):
    return np.sqrt(((a-b)**2).mean()) / np.sqrt((a**2).mean())

print(f"std(delta_2N)        = {delta_2N.std():.4e}")
print(f"std(delta_N)         = {delta_N.std():.4e}")
print(f"std(Psi_x 2N avg)    = {psix_2N_avg.std():.4e}")
print(f"std(Psi_x N)         = {psix_N.std():.4e}")
print(f"rel. RMS error in Psi_x (avg vs N) = {relerr(psix_2N_avg, psix_N):.4e}")
print(f"rel. RMS error in Psi_y (avg vs N) = {relerr(psiy_2N_avg, psiy_N):.4e}")

# for reference, same for delta itself (should be larger):
gx_2N = np.fft.ifftn(1j*KX*np.fft.fftn(delta_2N)).real
gx_N  = np.fft.ifftn(1j*KXc*np.fft.fftn(delta_N)).real
gx_2N_avg = gx_2N.reshape(N,2,N,2).mean(axis=(1,3))
print(f"rel. RMS error in d_x delta        = {relerr(gx_2N_avg, gx_N):.4e}  (for comparison)")

# ---- plots ----
fig, ax = plt.subplots(2, 3, figsize=(13, 8))

vmax = max(abs(psix_2N_avg).max(), abs(psix_N).max())
im = ax[0,0].imshow(psix_2N_avg, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,0].set_title(r'$\overline{\Psi_x^{(1)}[\delta_{2N}]}$ (fine, then restrict)')
plt.colorbar(im, ax=ax[0,0], fraction=0.045)

im = ax[0,1].imshow(psix_N, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,1].set_title(r'$\Psi_x^{(1)}[\delta_N]$ (restrict, then solve)')
plt.colorbar(im, ax=ax[0,1], fraction=0.045)

diff = psix_2N_avg - psix_N
dmax = max(abs(diff).max(), 1e-30)
im = ax[0,2].imshow(diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax, origin='lower')
ax[0,2].set_title(r'difference (LPT restriction error)')
plt.colorbar(im, ax=ax[0,2], fraction=0.045)

# quiver: sub-sample for clarity
step = max(1, N // 32)
xx = np.arange(N); yy = np.arange(N)
XX, YY = np.meshgrid(xx, yy, indexing='ij')
ax[1,0].quiver(XX[::step,::step], YY[::step,::step],
               psix_2N_avg[::step,::step], psiy_2N_avg[::step,::step],
               scale=None, color='C0')
ax[1,0].set_aspect('equal'); ax[1,0].set_title(r'$\bar\Psi^{(1)}$ from fine')

ax[1,1].quiver(XX[::step,::step], YY[::step,::step],
               psix_N[::step,::step], psiy_N[::step,::step],
               scale=None, color='C1')
ax[1,1].set_aspect('equal'); ax[1,1].set_title(r'$\Psi^{(1)}$ from restricted')

# 1D slices through the middle row
y0 = N // 2
ax[1,2].plot(psix_2N_avg[:, y0], label=r'$\bar\Psi_x$ (fine)', lw=1.5)
ax[1,2].plot(psix_N[:, y0], '--', label=r'$\Psi_x$ (restricted)', lw=1.5)
ax[1,2].set_xlabel('i'); ax[1,2].set_ylabel(r'$\Psi_x^{(1)}$')
ax[1,2].set_title(f'slice at y = {y0}'); ax[1,2].legend()

plt.tight_layout()
plt.savefig('restriction_psi1_2d.png', dpi=120)
print("wrote restriction_psi1_2d.png")
