"""Figure: same restriction comparisons but with a CDM-like P(k) from CLASS (z=200).

Shows that with a realistic CDM transfer function (P(k) ~ k^-3 ln^2 k at high k),
the restriction effects on delta itself are visible only near the coarse Nyquist,
and the 1LPT displacement and 2LPT source are essentially indistinguishable
between the fine-then-restrict and restrict-then-solve paths.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CLASS_PK = Path(__file__).resolve().parents[2] / 'data' / 'class_pk_z200_pk.dat'

rng = np.random.default_rng(42)
N, N2, L = 64, 128, 500.0   # L in Mpc/h, large enough that small-k modes have CDM peak

# --- load and interpolate CLASS P(k) ---
ktab, Ptab = np.loadtxt(CLASS_PK, comments='#', unpack=True)
def Pk_of_k(k):
    out = np.zeros_like(k)
    m = (k > 0) & (k >= ktab.min()) & (k <= ktab.max())
    out[m] = np.exp(np.interp(np.log(k[m]), np.log(ktab), np.log(Ptab)))
    return out

# --- generate fine field ---
kx = np.fft.fftfreq(N2, d=L/N2) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
Kmag = np.sqrt(KX**2 + KY**2)
Pk = Pk_of_k(Kmag)

white = rng.standard_normal((N2, N2))
dk = np.fft.fftn(white) * np.sqrt(Pk); dk[0,0] = 0.0
delta_2N = np.fft.ifftn(dk).real
delta_2N -= delta_2N.mean()
delta_N  = delta_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))

# --- coarse k grid ---
kc = np.fft.fftfreq(N, d=L/N) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')

def restrict(f): return f.reshape(N, 2, N, 2).mean(axis=(1, 3))

def psi1(delta, kxg, kyg):
    K2 = kxg**2 + kyg**2
    inv = np.where(K2 > 0, 1.0/K2, 0.0)
    dh = np.fft.fftn(delta)
    return (np.fft.ifftn(1j*kxg*inv*dh).real,
            np.fft.ifftn(1j*kyg*inv*dh).real)

def deformation(delta, kxg, kyg):
    K2 = kxg**2 + kyg**2
    inv = np.where(K2 > 0, 1.0/K2, 0.0)
    dh = np.fft.fftn(delta)
    dxx = np.fft.ifftn(-kxg*kxg*inv*dh).real
    dxy = np.fft.ifftn(-kxg*kyg*inv*dh).real
    dyy = np.fft.ifftn(-kyg*kyg*inv*dh).real
    return dxx, dxy, dyy

# fine-then-restrict
psix_f, _ = psi1(delta_2N, KX, KY)
psix_ref  = restrict(psix_f)
dxx_f, dxy_f, dyy_f = deformation(delta_2N, KX, KY)
S2_f   = dxx_f * dyy_f - dxy_f**2
S2_ref = restrict(S2_f)

# restrict-then-solve
psix_N, _ = psi1(delta_N, KXc, KYc)
dxx_N, dxy_N, dyy_N = deformation(delta_N, KXc, KYc)
S2_N = dxx_N * dyy_N - dxy_N**2

# input P(k) plot
fig, ax = plt.subplots(2, 3, figsize=(13, 8))

# (top-left) input P(k) vs k^-1 toy
kshow = np.logspace(np.log10(ktab.min()), np.log10(ktab.max()), 400)
Pshow = Pk_of_k(kshow)
ax[0,0].loglog(kshow, Pshow, 'k-', label='CLASS $P(k)$, $z=200$')
norm = Pshow[200] * kshow[200]
ax[0,0].loglog(kshow, norm/kshow, 'r--', label=r'toy $\propto k^{-1}$')
ax[0,0].axvline(np.pi*N/L,  color='C0', ls=':', label=r'$k_{Ny,N}$')
ax[0,0].axvline(np.pi*N2/L, color='C1', ls=':', label=r'$k_{Ny,2N}$')
ax[0,0].set_xlabel('k [h/Mpc]'); ax[0,0].set_ylabel('P(k)')
ax[0,0].legend(fontsize=8); ax[0,0].set_title('input spectrum')

# (top-mid, top-right) delta_2N and delta_N
vmax = max(abs(delta_2N).max(), abs(delta_N).max())
im = ax[0,1].imshow(delta_2N, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,1].set_title(r'$\delta_{2N}$'); plt.colorbar(im, ax=ax[0,1], fraction=0.045)
im = ax[0,2].imshow(delta_N,  cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
ax[0,2].set_title(r'$\delta_N$'); plt.colorbar(im, ax=ax[0,2], fraction=0.045)

# bottom row: differences for delta, Psi_x, S^(2)
diff_d = restrict(delta_2N) - delta_N  # zero by construction
diff_psi = psix_ref - psix_N
diff_s2  = S2_ref - S2_N

vmax = max(abs(restrict(delta_2N)).max(), 1e-30)
im = ax[1,0].imshow(restrict(delta_2N) - delta_N, cmap='RdBu_r', vmin=-vmax/10, vmax=vmax/10, origin='lower')
ax[1,0].set_title(r'$R\delta_{2N} - \delta_N$ (=0, sanity)')
plt.colorbar(im, ax=ax[1,0], fraction=0.045)

vmax = max(abs(psix_ref).max(), 1e-30)
im = ax[1,1].imshow(diff_psi, cmap='RdBu_r', vmin=-vmax/20, vmax=vmax/20, origin='lower')
ax[1,1].set_title(r'$\Psi_x$ error (paths diff)')
plt.colorbar(im, ax=ax[1,1], fraction=0.045)

vmax = max(abs(S2_ref).max(), 1e-30)
im = ax[1,2].imshow(diff_s2, cmap='RdBu_r', vmin=-vmax/5, vmax=vmax/5, origin='lower')
ax[1,2].set_title(r'$S^{(2)}$ error (paths diff)')
plt.colorbar(im, ax=ax[1,2], fraction=0.045)

plt.tight_layout()
plt.savefig('restriction_cdm.pdf')
print("wrote restriction_cdm.pdf")

def relerr(a, b):
    return np.sqrt(((a-b)**2).mean()) / np.sqrt((a**2).mean())
print(f"rel.RMS Psi_x error = {relerr(psix_ref, psix_N):.3e}")
print(f"rel.RMS S^(2) error = {relerr(S2_ref, S2_N):.3e}")
