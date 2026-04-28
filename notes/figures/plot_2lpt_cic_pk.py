"""2LPT particle generation + CIC P(k): a pedagogical IC pipeline.

Pipeline:
  1. Generate delta_2N on the fine grid via FFT-convolution
       delta = T(k) * mu, with mu white noise.
  2. Compute the 1LPT displacement Psi^(1) (Zel'dovich) on the fine grid
       Psi^(1)(k) = i k / k^2 * delta(k)
  3. Compute the 2LPT correction Psi^(2)
       S^(2) = Psi^(1)_{x,x} Psi^(1)_{y,y} - (Psi^(1)_{x,y})^2     (2D)
       Psi^(2)(k) = i k / k^2 * S^(2)(k)
       Particle displacement: x = q + Psi^(1) - (3/7) Psi^(2)
       (the -3/7 is the second-order growth ratio in EdS / flat LCDM)
  4. Particles at Lagrangian positions q (= fine-grid centres), one per
     fine cell.  Compute Eulerian positions x.
  5. Deposit particles onto the same fine grid via 2D CIC (cloud-in-cell):
     each particle's mass is distributed linearly to the 4 nearest cells.
  6. FFT the CIC density field, compute the radial-binned P(k).
  7. Deconvolve the CIC window W_CIC(k) = (sinc(k_x dx/2) sinc(k_y dx/2))^2
     to recover the underlying P(k).
  8. Compare measured (raw and deconvolved) with the input theory and
     with FFT(delta_2N) (the "no-particles" reference).
"""
import numpy as np
import matplotlib.pyplot as plt

# Setup matches the rest of the demo.
rng = np.random.default_rng(42)
N2, L = 128, 1.0
n_pk = -2.0
dx = L / N2

kx = np.fft.fftfreq(N2, d=dx) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
K2 = KX**2 + KY**2
T  = np.where(K2 > 0, K2**(0.25*n_pk), 0.0)            # T(k) = k^(n/2)
inv_K2 = np.where(K2 > 0, 1.0/K2, 0.0)

# 1. Generate delta_2N.
mu_k = np.fft.fftn(rng.standard_normal((N2, N2)))
delta_k = T * mu_k
delta_k[0, 0] = 0.0
delta = np.fft.ifftn(delta_k).real

# 2. 1LPT (Zel'dovich) displacement: Psi^(1)(k) = i k / k^2 * delta(k).
psi1_kx = 1j * KX * inv_K2 * delta_k
psi1_ky = 1j * KY * inv_K2 * delta_k
psi1_x  = np.fft.ifftn(psi1_kx).real
psi1_y  = np.fft.ifftn(psi1_ky).real

# 3. 2LPT correction: S^(2) from deformation tensor.
psi1_xx = np.fft.ifftn(1j*KX*np.fft.fftn(psi1_x)).real
psi1_xy = np.fft.ifftn(1j*KY*np.fft.fftn(psi1_x)).real
psi1_yy = np.fft.ifftn(1j*KY*np.fft.fftn(psi1_y)).real
S2 = psi1_xx * psi1_yy - psi1_xy**2
S2_k = np.fft.fftn(S2)
psi2_x = np.fft.ifftn(1j*KX*inv_K2*S2_k).real
psi2_y = np.fft.ifftn(1j*KY*inv_K2*S2_k).real

# Combine 1LPT + 2LPT.
disp_x = psi1_x - (3.0/7.0)*psi2_x
disp_y = psi1_y - (3.0/7.0)*psi2_y

# 4. Particles at Lagrangian positions = fine-cell centres.
q_axis = (np.arange(N2) + 0.5) * dx
QX, QY = np.meshgrid(q_axis, q_axis, indexing='ij')
x_part = (QX + disp_x) % L
y_part = (QY + disp_y) % L

# 5. 2D CIC deposit on fine grid (N2 x N2 cells).
# Each particle has unit mass.  Distribute to the 4 nearest cells.
def cic_deposit_2d(x, y, ngrid, box_size):
    """2D CIC mass assignment.  Returns a density field in units where
    the mean overdensity is zero (delta = rho/rho_bar - 1)."""
    cell = box_size / ngrid
    # Particle position in cell-index units; cell centres at 0.5, 1.5, ...
    fx = x / cell - 0.5
    fy = y / cell - 0.5
    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    dx_part = fx - i0
    dy_part = fy - j0
    i1 = i0 + 1
    j1 = j0 + 1
    # Periodic wrap.
    i0 %= ngrid; i1 %= ngrid
    j0 %= ngrid; j1 %= ngrid
    weights = [
        ((1-dx_part)*(1-dy_part), i0, j0),
        ((  dx_part)*(1-dy_part), i1, j0),
        ((1-dx_part)*(  dy_part), i0, j1),
        ((  dx_part)*(  dy_part), i1, j1),
    ]
    rho = np.zeros((ngrid, ngrid))
    for w, ii, jj in weights:
        np.add.at(rho, (ii, jj), w)
    rho_bar = rho.mean()
    return rho / rho_bar - 1.0

delta_cic = cic_deposit_2d(x_part.ravel(), y_part.ravel(), N2, L)

# 6. FFT the CIC density and bin radially.
def pma_pk(field, ngrid):
    fk = np.fft.fftn(field) / ngrid**2
    return np.abs(fk)**2

P_cic   = pma_pk(delta_cic, N2)
P_grid  = pma_pk(delta,     N2)            # fine-grid input delta (no particles)

# 7. Deconvolve CIC window.
# In 2D, W_CIC(k) = sinc(k_x dx / 2) sinc(k_y dx / 2)  (per axis).
# np.sinc(x) = sin(pi x)/(pi x), so np.sinc(k dx / (2 pi)) gives sin(k dx / 2)/(k dx / 2).
W_CIC = np.sinc(KX*dx/(2*np.pi)) * np.sinc(KY*dx/(2*np.pi))
P_cic_deconv = P_cic / np.where(W_CIC**2 > 1e-8, W_CIC**2, 1.0)

# Radial mean.
def radial_mean(P, K, nbins=22, kmax=None):
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

K_mag = np.sqrt(K2)
k_bar, P_grid_bar = radial_mean(P_grid, K_mag)
_,     P_cic_bar  = radial_mean(P_cic,  K_mag)
_,     P_dec_bar  = radial_mean(P_cic_deconv, K_mag)
P_theory = (k_bar**n_pk) / N2**2

# 8. Plot.
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# Left: spectra overlay.
ax[0].loglog(k_bar, P_theory, 'k-', lw=1, label=r'theory $\propto k^{n}$')
ax[0].loglog(k_bar, P_grid_bar, 'o', color='gray', alpha=0.7, ms=5,
             label=r'$P[\delta_{\rm grid}]$ (input on grid)')
ax[0].loglog(k_bar, P_cic_bar, 's', color='C1', alpha=0.7, ms=5,
             label=r'$P[\delta_{\rm CIC}]$ (raw, after CIC deposit)')
ax[0].loglog(k_bar, P_dec_bar, 'd', color='C0', alpha=0.7, ms=5,
             label=r'$P[\delta_{\rm CIC}] / W_{\rm CIC}^2$ (deconvolved)')
ax[0].axvline(np.pi/dx, color='gray', ls=':', label=r'fine Nyquist')
ax[0].set_xlabel(r'$|k|$'); ax[0].set_ylabel(r'$P(k)$')
ax[0].set_title('2LPT particles, deposited via CIC')
ax[0].legend(fontsize=8)

# Right: ratio to theory.
ax[1].plot(k_bar, P_grid_bar  / P_theory, 'o-', color='gray',
           label=r'grid input / theory')
ax[1].plot(k_bar, P_cic_bar   / P_theory, 's-', color='C1',
           label=r'CIC raw / theory')
ax[1].plot(k_bar, P_dec_bar   / P_theory, 'd-', color='C0',
           label=r'CIC deconvolved / theory')
ax[1].axhline(1, color='gray', lw=0.5)
ax[1].axvline(np.pi/dx, color='gray', ls=':', label=r'fine Nyquist')
ax[1].set_xscale('log')
ax[1].set_xlabel(r'$|k|$'); ax[1].set_ylabel(r'ratio')
ax[1].set_ylim(0, 2.0)
ax[1].set_title('ratio to input theory')
ax[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('lpt_cic_pk.pdf')
print('wrote lpt_cic_pk.pdf')

# Diagnostics.
print(f"\n=== 2LPT + CIC P(k) diagnostics ===")
print(f"max |displacement| / cell size: {max(abs(disp_x).max(), abs(disp_y).max())/dx:.3f}")
print(f"P_grid / theory (median):  {np.median(P_grid_bar / P_theory):.3f}")
print(f"P_CIC raw / theory (median): {np.median(P_cic_bar / P_theory):.3f}")
print(f"P_CIC deconv / theory (median): {np.median(P_dec_bar / P_theory):.3f}")
