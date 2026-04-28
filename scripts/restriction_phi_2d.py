"""
restriction_phi_2d.py — three ways to get Psi^(1) on a coarse grid:

  (1) reference: Psi_2N = -grad phi_2N on the fine grid, then 2x2 block average
  (2) restrict-delta:   delta_N = restrict(delta_2N); solve Poisson on coarse;
                        Psi_N = -grad phi_N
  (3) restrict-phi:     phi_2N = -inv-Laplacian(delta_2N) on the fine grid;
                        phi_N' = restrict(phi_2N); Psi_N' = -grad phi_N' on coarse

The principal-mode contribution is identical in all three. The differences live
in the alias contributions and, more importantly, in self-consistency:

  Path (3) does NOT satisfy nabla^2 phi_N' = -delta_N on the coarse grid.
  Equivalently, the "effective density" -nabla^2 phi_N' differs from delta_N.

That's the LPT failure mode of restricting the potential.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
N    = 128
N2   = 2 * N
L    = 1.0
n_pk = -1.0   # try 0 or +1 to make alias effects more visible

# ---- fine field, P(k) propto k^n ----
kx = np.fft.fftfreq(N2, d=L/N2) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
K2_f   = KX**2 + KY**2
Pk     = np.where(K2_f > 0, K2_f**(0.5*n_pk), 0.0)

white  = rng.standard_normal((N2, N2))
dk     = np.fft.fftn(white) * np.sqrt(Pk)
dk[0, 0] = 0.0
delta_2N = np.fft.ifftn(dk).real
delta_2N -= delta_2N.mean()

# coarse k-grid
kc = np.fft.fftfreq(N, d=L/N) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')
K2_c     = KXc**2 + KYc**2

inv_K2_f = np.where(K2_f > 0, 1.0/K2_f, 0.0)
inv_K2_c = np.where(K2_c > 0, 1.0/K2_c, 0.0)

def restrict(field):
    return field.reshape(N, 2, N, 2).mean(axis=(1, 3))

def gradient_2d(field, kx_grid, ky_grid):
    fk = np.fft.fftn(field)
    gx = np.fft.ifftn(1j * kx_grid * fk).real
    gy = np.fft.ifftn(1j * ky_grid * fk).real
    return gx, gy

def laplacian_2d(field, kx_grid, ky_grid):
    fk = np.fft.fftn(field)
    return np.fft.ifftn(-(kx_grid**2 + ky_grid**2) * fk).real

# ---------- (1) reference: phi and Psi on the fine grid, then restrict ----------
phi_2N    = np.fft.ifftn(-np.fft.fftn(delta_2N) * inv_K2_f).real
psix_2N, psiy_2N = gradient_2d(-phi_2N, KX, KY)   # Psi = -grad phi
psix_ref  = restrict(psix_2N)
psiy_ref  = restrict(psiy_2N)

# ---------- (2) restrict-delta: solve Poisson on coarse ----------
delta_N    = restrict(delta_2N)
phi_N_rd   = np.fft.ifftn(-np.fft.fftn(delta_N) * inv_K2_c).real
psix_rd, psiy_rd = gradient_2d(-phi_N_rd, KXc, KYc)

# ---------- (3) restrict-phi: differentiate on coarse ----------
phi_N_rp   = restrict(phi_2N)
psix_rp, psiy_rp = gradient_2d(-phi_N_rp, KXc, KYc)

# ---------- self-consistency check: -laplacian(phi_N_rp) vs delta_N ----------
delta_eff  = -laplacian_2d(phi_N_rp, KXc, KYc)

# ---------- diagnostics ----------
def relerr(a, b):
    return np.sqrt(((a-b)**2).mean()) / np.sqrt((a**2).mean())

print(f"std(delta_2N)            = {delta_2N.std():.4e}")
print(f"std(delta_N)             = {delta_N.std():.4e}")
print(f"std(delta_eff = -lap phi_N')= {delta_eff.std():.4e}")
print()
print("relative RMS error vs the fine-then-restrict reference:")
print(f"  Psi_x, restrict-delta = {relerr(psix_ref, psix_rd):.4e}")
print(f"  Psi_x, restrict-phi   = {relerr(psix_ref, psix_rp):.4e}")
print()
print(f"Poisson inconsistency on coarse grid:")
print(f"  rel. RMS(delta_eff - delta_N) / RMS(delta_N) = {relerr(delta_N, delta_eff):.4e}")

# ---------- plots ----------
fig, ax = plt.subplots(2, 3, figsize=(13, 8))

vmax = max(abs(psix_ref).max(), abs(psix_rd).max(), abs(psix_rp).max())
im = ax[0,0].imshow(psix_ref, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,0].set_title(r'(1) reference: $\overline{\Psi_x}$ from fine'); plt.colorbar(im, ax=ax[0,0], fraction=0.045)
im = ax[0,1].imshow(psix_rd,  cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,1].set_title(r'(2) restrict $\delta$, then Poisson'); plt.colorbar(im, ax=ax[0,1], fraction=0.045)
im = ax[0,2].imshow(psix_rp,  cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,2].set_title(r'(3) restrict $\phi$, then $-\nabla\phi$'); plt.colorbar(im, ax=ax[0,2], fraction=0.045)

# error maps
emax = max(abs(psix_ref - psix_rd).max(), abs(psix_ref - psix_rp).max(), 1e-30)
im = ax[1,0].imshow(psix_ref - psix_rd, cmap='RdBu_r', vmin=-emax, vmax=emax, origin='lower')
ax[1,0].set_title(r'error: (1) - (2)'); plt.colorbar(im, ax=ax[1,0], fraction=0.045)
im = ax[1,1].imshow(psix_ref - psix_rp, cmap='RdBu_r', vmin=-emax, vmax=emax, origin='lower')
ax[1,1].set_title(r'error: (1) - (3)'); plt.colorbar(im, ax=ax[1,1], fraction=0.045)

# the punchline: delta_eff vs delta_N
dmax = max(abs(delta_N).max(), abs(delta_eff).max())
im = ax[1,2].imshow(delta_eff - delta_N, cmap='RdBu_r',
                    vmin=-dmax, vmax=dmax, origin='lower')
ax[1,2].set_title(r'$-\nabla^2\phi_N^{\rm (restrict\,\phi)} - \delta_N$'+'\n(Poisson inconsistency)')
plt.colorbar(im, ax=ax[1,2], fraction=0.045)

plt.tight_layout()
plt.savefig('restriction_phi_2d.png', dpi=120)
print("wrote restriction_phi_2d.png")

# extra: 1D slices through y = N//2 for Psi_x
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
y0 = N // 2
ax2[0].plot(psix_ref[:, y0],          label='(1) reference', lw=1.5)
ax2[0].plot(psix_rd [:, y0], '--',    label=r'(2) restrict $\delta$', lw=1.5)
ax2[0].plot(psix_rp [:, y0], ':',     label=r'(3) restrict $\phi$',   lw=1.5)
ax2[0].set_xlabel('i'); ax2[0].set_ylabel(r'$\Psi_x^{(1)}$'); ax2[0].legend()
ax2[0].set_title('Psi_x slice')

ax2[1].plot(delta_N  [:, y0],       label=r'$\delta_N$ (true restricted density)', lw=1.5)
ax2[1].plot(delta_eff[:, y0], '--', label=r'$-\nabla^2\phi_N^{\rm (restrict\,\phi)}$', lw=1.5)
ax2[1].set_xlabel('i'); ax2[1].set_ylabel(r'$\delta$'); ax2[1].legend()
ax2[1].set_title('Poisson check')

plt.tight_layout()
plt.savefig('restriction_phi_2d_slices.png', dpi=120)
print("wrote restriction_phi_2d_slices.png")
