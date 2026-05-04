"""Figure: three coarse-grid paths for Psi_x^(1) and Poisson inconsistency of restricting phi."""
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

def restrict(f):    return f.reshape(N, 2, N, 2).mean(axis=(1, 3))
def grad(f, kxg, kyg):
    fk = np.fft.fftn(f)
    return np.fft.ifftn(1j*kxg*fk).real, np.fft.ifftn(1j*kyg*fk).real
def lap(f, kxg, kyg):
    return np.fft.ifftn(-(kxg**2+kyg**2) * np.fft.fftn(f)).real

# (1) reference
phi_2N = np.fft.ifftn(-np.fft.fftn(delta_2N) * inv_K2_f).real
psix_2N, _ = grad(-phi_2N, KX, KY)
psix_ref = restrict(psix_2N)

# (2) restrict delta
delta_N  = restrict(delta_2N)
phi_N_rd = np.fft.ifftn(-np.fft.fftn(delta_N) * inv_K2_c).real
psix_rd, _ = grad(-phi_N_rd, KXc, KYc)

# (3) restrict phi
phi_N_rp = restrict(phi_2N)
psix_rp, _ = grad(-phi_N_rp, KXc, KYc)
delta_eff = lap(phi_N_rp, KXc, KYc)

fig, ax = plt.subplots(2, 3, figsize=(13, 8))

vmax = max(abs(psix_ref).max(), abs(psix_rd).max(), abs(psix_rp).max())
for a, fld, title in zip(
        ax[0], (psix_ref, psix_rd, psix_rp),
        (r'(1) fine $\Psi$, then restrict' + '\n' + r'($\mathcal{R}\,\Psi_x^{(1)}[\delta_{2N}]$)',
         r'(2) restrict $\delta$, then solve' + '\n' + r'($\Psi_x^{(1)}[\mathcal{R}\,\delta_{2N}]$)',
         r'(3) restrict $\phi$, then $-\nabla\phi$' + '\n' + r'($-\partial_x(\mathcal{R}\,\phi^{(1)}_{2N})$)')):
    im = a.imshow(fld, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
    a.set_title(title); plt.colorbar(im, ax=a, fraction=0.045)

emax = max(abs(psix_ref - psix_rd).max(), abs(psix_ref - psix_rp).max(), 1e-30)
im = ax[1,0].imshow(psix_ref - psix_rd, cmap='RdBu_r', vmin=-emax, vmax=emax, origin='lower')
ax[1,0].set_title('error: (1) - (2)'); plt.colorbar(im, ax=ax[1,0], fraction=0.045)
im = ax[1,1].imshow(psix_ref - psix_rp, cmap='RdBu_r', vmin=-emax, vmax=emax, origin='lower')
ax[1,1].set_title('error: (1) - (3)'); plt.colorbar(im, ax=ax[1,1], fraction=0.045)

dmax = max(abs(delta_N).max(), abs(delta_eff).max())
im = ax[1,2].imshow(delta_eff - delta_N, cmap='RdBu_r', vmin=-dmax, vmax=dmax, origin='lower')
ax[1,2].set_title(r'$\nabla^2(\mathcal{R}\,\phi^{(1)}_{2N}) - \delta_N$' + '\n(Poisson inconsistency)')
plt.colorbar(im, ax=ax[1,2], fraction=0.045)

fig.suptitle(rf'Power-law input $P(k)\propto k^{{n}}$ with $n={n_pk:g}$')
plt.tight_layout()
plt.savefig('restriction_phi.pdf')
print("wrote restriction_phi.pdf")
