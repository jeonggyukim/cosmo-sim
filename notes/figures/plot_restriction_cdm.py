"""Figure: 3D CDM restriction comparison of constructions (1), (2), (3).

Constructions follow the §5 table:
  (1) fine Psi/S2 then restrict      -- reference
  (2) restrict delta then solve      -- amplifies aliases via 1/k_c instead of 1/k'
  (3) restrict phi then -grad phi    -- keeps 1/(k')^2 suppression but Poisson-inconsistent

Layout: 2 rows (Psi_x^(1) on top, S^(2) on bottom). Each row has, left to right:
the reference field (1) on a z-slice, the residual (1)-(2) z-slice, the residual
(1)-(3) z-slice, and a histogram of the per-cell relative errors of (2) and (3)
versus (1) (computed over the full 3D volume, not the slice).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from icpipe import LinearTheory

CLASS_PK = Path(__file__).resolve().parents[2] / 'data' / 'class_pk_z200_pk.dat'

rng = np.random.default_rng(42)
N, N2, L = 64, 128, 500.0
th = LinearTheory.from_class(CLASS_PK, z=200)
Pk_of_k = th.Pk

kx = np.fft.fftfreq(N2, d=L/N2) * 2*np.pi
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
Pk = Pk_of_k(Kmag)

white = rng.standard_normal((N2, N2, N2))
dk = np.fft.fftn(white) * np.sqrt(Pk); dk[0,0,0] = 0.0
delta_2N = np.fft.ifftn(dk).real
delta_2N -= delta_2N.mean()

kc = np.fft.fftfreq(N, d=L/N) * 2*np.pi
KXc, KYc, KZc = np.meshgrid(kc, kc, kc, indexing='ij')

def restrict(f):
    return f.reshape(N,2,N,2,N,2).mean(axis=(1,3,5))

delta_N = restrict(delta_2N)

def solve_phi(delta, kxg, kyg, kzg):
    K2 = kxg**2 + kyg**2 + kzg**2
    inv = np.where(K2 > 0, 1.0/K2, 0.0)
    return np.fft.ifftn(-np.fft.fftn(delta) * inv).real

def psi_x_from_phi(phi, kxg):
    return np.fft.ifftn(-1j*kxg * np.fft.fftn(phi)).real

def deformation_for_psi(phi, kxg, kyg, kzg):
    """d_i Psi_j = -d_i d_j phi -> (+k_i k_j) phi_hat."""
    phih = np.fft.fftn(phi)
    return tuple(np.fft.ifftn(ka*kb * phih).real
                 for (ka, kb) in [(kxg,kxg),(kyg,kyg),(kzg,kzg),
                                  (kxg,kyg),(kxg,kzg),(kyg,kzg)])

def S2_3d(p):
    pxx, pyy, pzz, pxy, pxz, pyz = p
    return (pxx*pyy - pxy**2) + (pxx*pzz - pxz**2) + (pyy*pzz - pyz**2)

# (1) fine phi, then restrict the LPT objects
phi_2N = solve_phi(delta_2N, KX, KY, KZ)
psix_f = psi_x_from_phi(phi_2N, KX)
defo_f = deformation_for_psi(phi_2N, KX, KY, KZ)
S2_f   = S2_3d(defo_f)
psix_1 = restrict(psix_f)
S2_1   = restrict(S2_f)
del psix_f, defo_f, S2_f

# (2) restrict delta, then solve on coarse grid
phi_N_rd = solve_phi(delta_N, KXc, KYc, KZc)
psix_2   = psi_x_from_phi(phi_N_rd, KXc)
defo_2   = deformation_for_psi(phi_N_rd, KXc, KYc, KZc)
S2_2     = S2_3d(defo_2)
del defo_2

# (3) restrict phi, then take coarse-grid gradients
phi_N_rp = restrict(phi_2N)
psix_3   = psi_x_from_phi(phi_N_rp, KXc)
defo_3   = deformation_for_psi(phi_N_rp, KXc, KYc, KZc)
S2_3     = S2_3d(defo_3)
del defo_3, phi_2N

sigma_psi = psix_1.std()
sigma_s2  = S2_1.std()
slc = N // 2

fig, ax = plt.subplots(2, 4, figsize=(17, 8))

def show_map(a, img, title):
    v = max(abs(img).max(), 1e-30)
    im = a.imshow(img, cmap='RdBu_r', vmin=-v, vmax=v, origin='lower')
    a.set_title(title, fontsize=10)
    plt.colorbar(im, ax=a, fraction=0.045)

def hist_panel(a, e12, e13, sigma_label, field_label):
    rms_12 = np.sqrt((e12**2).mean()) / sigma_label
    rms_13 = np.sqrt((e13**2).mean()) / sigma_label
    bins = 80
    a.hist(e12.ravel()/sigma_label, bins=bins, histtype='step', linewidth=1.4,
           color='C0', label=f'(1)$-$(2): RMS = {rms_12:.1%}')
    a.hist(e13.ravel()/sigma_label, bins=bins, histtype='step', linewidth=1.4,
           color='C1', label=f'(1)$-$(3): RMS = {rms_13:.1%}')
    a.set_xlabel(f'residual / $\\sigma_{{{field_label}}}$')
    a.set_ylabel('number of coarse-grid cells')
    a.legend(fontsize=8)
    a.set_title('per-cell relative error', fontsize=10)

# top row: Psi_x
show_map(ax[0,0], psix_1[:,:,slc]/sigma_psi,
         r'(1) reference $\mathcal{R}\,\Psi_x^{(1)}/\sigma$')
show_map(ax[0,1], (psix_1-psix_2)[:,:,slc]/sigma_psi,
         r'(1)$-$(2): residual$/\sigma_{\Psi_x^{(1)}}$')
show_map(ax[0,2], (psix_1-psix_3)[:,:,slc]/sigma_psi,
         r'(1)$-$(3): residual$/\sigma_{\Psi_x^{(1)}}$')
hist_panel(ax[0,3], psix_1-psix_2, psix_1-psix_3, sigma_psi, r'\Psi_x^{(1)}')

# bottom row: S^(2)
show_map(ax[1,0], S2_1[:,:,slc]/sigma_s2,
         r'(1) reference $\mathcal{R}\,S^{(2)}/\sigma$')
show_map(ax[1,1], (S2_1-S2_2)[:,:,slc]/sigma_s2,
         r'(1)$-$(2): residual$/\sigma_{S^{(2)}}$')
show_map(ax[1,2], (S2_1-S2_3)[:,:,slc]/sigma_s2,
         r'(1)$-$(3): residual$/\sigma_{S^{(2)}}$')
hist_panel(ax[1,3], S2_1-S2_2, S2_1-S2_3, sigma_s2, r'S^{(2)}')

plt.tight_layout()
plt.savefig('restriction_cdm.pdf')
print('wrote restriction_cdm.pdf')

def relrms(a, b, ref):
    return np.sqrt(((a-b)**2).mean()) / np.sqrt((ref**2).mean())

print(f'3D CDM, N2={N2}, L={L} Mpc/h')
print(f'  Psi_x:  rel.RMS (1)-(2) = {relrms(psix_1, psix_2, psix_1):.3e}')
print(f'          rel.RMS (1)-(3) = {relrms(psix_1, psix_3, psix_1):.3e}')
print(f'  S^(2):  rel.RMS (1)-(2) = {relrms(S2_1,   S2_2,   S2_1  ):.3e}')
print(f'          rel.RMS (1)-(3) = {relrms(S2_1,   S2_3,   S2_1  ):.3e}')
