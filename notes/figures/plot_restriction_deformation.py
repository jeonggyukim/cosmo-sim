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

fig, ax = plt.subplots(2, 4, figsize=(17, 8))

def panel_row(row_axes, field_a, field_b, name_a, name_b, hist_label, hist_title_quantity):
    vmax = max(abs(field_a).max(), abs(field_b).max())
    im = row_axes[0].imshow(field_a, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
    row_axes[0].set_title(name_a)
    plt.colorbar(im, ax=row_axes[0], fraction=0.045)

    im = row_axes[1].imshow(field_b, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
    row_axes[1].set_title(name_b)
    plt.colorbar(im, ax=row_axes[1], fraction=0.045)

    diff = field_a - field_b
    dmax = max(abs(diff).max(), 1e-30)
    im = row_axes[2].imshow(diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax, origin='lower')
    row_axes[2].set_title('(a) $-$ (b)\n(colour scale rescaled)')
    plt.colorbar(im, ax=row_axes[2], fraction=0.045)

    rms_a = np.sqrt((field_a**2).mean())
    relerr = diff.ravel() / rms_a
    rms_rel = np.sqrt((relerr**2).mean())
    maxabs = np.max(np.abs(relerr))
    row_axes[3].hist(relerr, bins=60, color='C0', edgecolor='k', linewidth=0.3)
    row_axes[3].set_xlabel(hist_label)
    row_axes[3].set_ylabel('number of coarse-grid cells')
    row_axes[3].set_title(f'per-cell relative error of (b) vs (a)\n'
                          f'for {hist_title_quantity}  '
                          f'(RMS = {rms_rel:.1%}, max = {maxabs:.1%})')
    return rms_rel, maxabs

# top row: deformation tensor diagonal d_x Psi_x.
rms_dxx, max_dxx = panel_row(
    ax[0],
    dxx_ref, dxx_N,
    r'(a) $\mathcal{R}\,\partial_x \Psi_x^{(1)}[\delta_{2N}]$',
    r'(b) $\partial_x \Psi_x^{(1)}[\mathcal{R}\,\delta_{2N}]$',
    r'$[\,(\partial_x\Psi_x)_{(a)} - (\partial_x\Psi_x)_{(b)}\,]'
    r'\,/\,\mathrm{RMS}((\partial_x\Psi_x)_{(a)})$',
    r'$\partial_x\Psi_x^{(1)}$',
)

# bottom row: 2LPT source S^(2).
rms_S2, max_S2 = panel_row(
    ax[1],
    S2_ref, S2_N,
    r'(a) $\mathcal{R}\,S^{(2)}[\delta_{2N}]$',
    r'(b) $S^{(2)}[\mathcal{R}\,\delta_{2N}]$',
    r'$[\,S^{(2)}_{(a)} - S^{(2)}_{(b)}\,]\,/\,\mathrm{RMS}(S^{(2)}_{(a)})$',
    r'$S^{(2)}$',
)

fig.suptitle(rf'Power-law input $P(k)\propto k^{{n}}$ with $n={n_pk:g}$')
plt.tight_layout()
plt.savefig('restriction_deformation.pdf')
print('wrote restriction_deformation.pdf')
print(f'rms_rel(d_x Psi_x) = {rms_dxx:.4%}, max = {max_dxx:.4%}')
print(f'rms_rel(S^(2))     = {rms_S2:.4%}, max = {max_S2:.4%}')
