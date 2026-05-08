"""Figure: residual power vs k_c against the fine-grid truth.

For each of Psi_x^(1) and S^(2), compute the per-mode residual

    Delta_(k)(k_c) = hat[Psi^(k)_N](k_c) - hat[Psi^fine_2N](k_c)

at every principal coarse mode k_c (i.e. inside the coarse Brillouin zone).
The fine-grid truth is the LPT object computed on the 2N grid; we sample its
Fourier amplitude at each coarse wavevector. Bin radially in k and plot

    P_resid(k) = < |Delta|^2 >_bin   /   P_truth(k),

i.e. the fractional residual power per mode for each construction.
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
Kmag_c = np.sqrt(KXc**2 + KYc**2 + KZc**2)

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
    phih = np.fft.fftn(phi)
    return tuple(np.fft.ifftn(ka*kb * phih).real
                 for (ka, kb) in [(kxg,kxg),(kyg,kyg),(kzg,kzg),
                                  (kxg,kyg),(kxg,kzg),(kyg,kzg)])

def S2_3d(p):
    pxx, pyy, pzz, pxy, pxz, pyz = p
    return (pxx*pyy - pxy**2) + (pxx*pzz - pxz**2) + (pyy*pzz - pyz**2)

# fine-grid truth
phi_2N = solve_phi(delta_2N, KX, KY, KZ)
psix_truth_fine = psi_x_from_phi(phi_2N, KX)
defo_truth_fine = deformation_for_psi(phi_2N, KX, KY, KZ)
S2_truth_fine = S2_3d(defo_truth_fine)
del defo_truth_fine

# (1) restrict the LPT objects from fine
psix_1 = restrict(psix_truth_fine)
S2_1   = restrict(S2_truth_fine)

# (2) restrict delta, then solve
phi_N_rd = solve_phi(delta_N, KXc, KYc, KZc)
psix_2 = psi_x_from_phi(phi_N_rd, KXc)
defo_2 = deformation_for_psi(phi_N_rd, KXc, KYc, KZc)
S2_2 = S2_3d(defo_2)
del defo_2

# (3) restrict phi, take coarse gradients
phi_N_rp = restrict(phi_2N)
psix_3 = psi_x_from_phi(phi_N_rp, KXc)
defo_3 = deformation_for_psi(phi_N_rp, KXc, KYc, KZc)
S2_3 = S2_3d(defo_3)
del defo_3

# extract truth at coarse wavevectors
def truth_hat_on_coarse_grid(field_fine):
    """Fine-grid Fourier amplitude sampled at the coarse Brillouin zone wavevectors,
    normalised the same way as `hat()` above so coarse and fine amplitudes are
    directly comparable at matching physical wavevectors.
    """
    fh = np.fft.fftn(field_fine) / field_fine.size
    int_freq = (np.fft.fftfreq(N, d=1.0)*N).astype(int)   # signed coarse freqs
    fine_idx = int_freq % N2
    out = np.zeros((N,N,N), dtype=complex)
    for i,ix in enumerate(fine_idx):
        for j,iy in enumerate(fine_idx):
            for k,iz in enumerate(fine_idx):
                out[i,j,k] = fh[ix, iy, iz]
    return out

psix_truth_hat_at_kc = truth_hat_on_coarse_grid(psix_truth_fine)
S2_truth_hat_at_kc   = truth_hat_on_coarse_grid(S2_truth_fine)

def hat(f):
    # Normalise so that hat[k] is the continuous Fourier amplitude (independent of grid size):
    # numpy's fftn returns hat = sum_x f(x) exp(-i k x) = N_total * <f exp(-i k x)>.
    # Dividing by N_total puts the coarse and fine FFTs on the same physical scale.
    return np.fft.fftn(f) / f.size

# Fourier amplitudes of constructions (already on coarse grid)
def radial_bin_power(quantity_squared, kmag, kbins):
    P = np.zeros(len(kbins)-1)
    for i in range(len(kbins)-1):
        m = (kmag >= kbins[i]) & (kmag < kbins[i+1])
        if m.any():
            P[i] = quantity_squared[m].mean()
    return P

# bin edges out to coarse Nyquist
k_Ny_c = np.pi*N/L
kbins = np.linspace(0, k_Ny_c, 24)
kmid = 0.5*(kbins[:-1]+kbins[1:])

def fractional_residual_power(coarse_field, truth_hat_at_kc):
    diff_hat = hat(coarse_field) - truth_hat_at_kc
    num = radial_bin_power(np.abs(diff_hat)**2, Kmag_c, kbins)
    den = radial_bin_power(np.abs(truth_hat_at_kc)**2, Kmag_c, kbins)
    return np.sqrt(np.where(den > 0, num/den, 0.0))

frac_psi_1 = fractional_residual_power(psix_1, psix_truth_hat_at_kc)
frac_psi_2 = fractional_residual_power(psix_2, psix_truth_hat_at_kc)
frac_psi_3 = fractional_residual_power(psix_3, psix_truth_hat_at_kc)
frac_S2_1  = fractional_residual_power(S2_1,   S2_truth_hat_at_kc)
frac_S2_2  = fractional_residual_power(S2_2,   S2_truth_hat_at_kc)
frac_S2_3  = fractional_residual_power(S2_3,   S2_truth_hat_at_kc)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

ax[0].plot(kmid, frac_psi_1, 'C0-',  label='(1) fine $\\Psi$, then restrict')
ax[0].plot(kmid, frac_psi_2, 'C1--', label='(2) restrict $\\delta$, then solve')
ax[0].plot(kmid, frac_psi_3, 'C2-.', label='(3) restrict $\\phi$, then $-\\nabla\\phi$')
ax[0].set_xlabel(r'$k_c$ [h/Mpc]')
ax[0].set_ylabel(r'$\sqrt{\langle|\Delta|^2\rangle/\langle|\hat\Psi^{\rm fine}|^2\rangle}$')
ax[0].set_title(r'$\Psi_x^{(1)}$: residual vs fine truth')
ax[0].axvline(k_Ny_c, color='k', ls=':', lw=0.8)
ax[0].legend(fontsize=8)

ax[1].plot(kmid, frac_S2_1, 'C0-',  label='(1)')
ax[1].plot(kmid, frac_S2_2, 'C1--', label='(2)')
ax[1].plot(kmid, frac_S2_3, 'C2-.', label='(3)')
ax[1].set_xlabel(r'$k_c$ [h/Mpc]')
ax[1].set_ylabel(r'$\sqrt{\langle|\Delta|^2\rangle/\langle|\hat S^{(2),\rm fine}|^2\rangle}$')
ax[1].set_title(r'$S^{(2)}$: residual vs fine truth')
ax[1].axvline(k_Ny_c, color='k', ls=':', lw=0.8)
ax[1].legend(fontsize=8)

for a in ax:
    a.set_xscale('log')
    a.set_yscale('log')

plt.tight_layout()
plt.savefig('restriction_cdm_truth.pdf')
print('wrote restriction_cdm_truth.pdf')

# integrated RMS over the coarse Brillouin zone
def total_relrms(coarse_field, truth_hat_at_kc):
    diff_hat = hat(coarse_field) - truth_hat_at_kc
    num = (np.abs(diff_hat)**2).sum()
    den = (np.abs(truth_hat_at_kc)**2).sum()
    return np.sqrt(num/den)

print(f'Integrated rel.RMS vs fine truth (over coarse Brillouin zone):')
print(f'  Psi_x:  (1)={total_relrms(psix_1, psix_truth_hat_at_kc):.3e}  '
      f'(2)={total_relrms(psix_2, psix_truth_hat_at_kc):.3e}  '
      f'(3)={total_relrms(psix_3, psix_truth_hat_at_kc):.3e}')
print(f'  S^(2):  (1)={total_relrms(S2_1,   S2_truth_hat_at_kc):.3e}  '
      f'(2)={total_relrms(S2_2,   S2_truth_hat_at_kc):.3e}  '
      f'(3)={total_relrms(S2_3,   S2_truth_hat_at_kc):.3e}')
