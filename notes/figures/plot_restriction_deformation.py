"""Figure: deformation tensor d_i Psi_j and 2LPT source under restriction."""
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

kc = np.fft.fftfreq(N, d=L/N) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')

def deformation(delta, kxg, kyg):
    """Return d_i Psi_j^(1) for i,j in {x,y}: kernel -k_i k_j / k^2 acting on delta."""
    K2g = kxg**2 + kyg**2
    inv = np.where(K2g > 0, 1.0/K2g, 0.0)
    dh  = np.fft.fftn(delta)
    dxx = np.fft.ifftn(-kxg*kxg*inv*dh).real
    dxy = np.fft.ifftn(-kxg*kyg*inv*dh).real
    dyy = np.fft.ifftn(-kyg*kyg*inv*dh).real
    return dxx, dxy, dyy

def restrict(f): return f.reshape(N, 2, N, 2).mean(axis=(1, 3))

# --- fine, then restrict ---
dxx_f, dxy_f, dyy_f = deformation(delta_2N, KX, KY)
dxx_ref = restrict(dxx_f); dxy_ref = restrict(dxy_f); dyy_ref = restrict(dyy_f)
# 2LPT source in 2D: S2 = Psi_xx Psi_yy - Psi_xy^2
S2_f   = dxx_f * dyy_f - dxy_f**2
S2_ref = restrict(S2_f)

# --- restrict, then solve ---
dxx_N, dxy_N, dyy_N = deformation(delta_N, KXc, KYc)
S2_N = dxx_N * dyy_N - dxy_N**2

fig, ax = plt.subplots(2, 3, figsize=(13, 8))

vmax = max(abs(dxx_ref).max(), abs(dxx_N).max())
im = ax[0,0].imshow(dxx_ref, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,0].set_title(r'$\overline{\partial_x \Psi_x^{(1)}[\delta_{2N}]}$')
plt.colorbar(im, ax=ax[0,0], fraction=0.045)
im = ax[0,1].imshow(dxx_N, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,1].set_title(r'$\partial_x \Psi_x^{(1)}[\delta_N]$')
plt.colorbar(im, ax=ax[0,1], fraction=0.045)
diff = dxx_ref - dxx_N
dmax = max(abs(diff).max(), 1e-30)
im = ax[0,2].imshow(diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax, origin='lower')
ax[0,2].set_title(r'difference')
plt.colorbar(im, ax=ax[0,2], fraction=0.045)

vmax = max(abs(S2_ref).max(), abs(S2_N).max())
im = ax[1,0].imshow(S2_ref, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[1,0].set_title(r'$\overline{S^{(2)}[\delta_{2N}]}$')
plt.colorbar(im, ax=ax[1,0], fraction=0.045)
im = ax[1,1].imshow(S2_N, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[1,1].set_title(r'$S^{(2)}[\delta_N]$')
plt.colorbar(im, ax=ax[1,1], fraction=0.045)
diff = S2_ref - S2_N
dmax = max(abs(diff).max(), 1e-30)
im = ax[1,2].imshow(diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax, origin='lower')
ax[1,2].set_title(r'difference')
plt.colorbar(im, ax=ax[1,2], fraction=0.045)

plt.tight_layout()
plt.savefig('restriction_deformation.pdf')
print("wrote restriction_deformation.pdf")
