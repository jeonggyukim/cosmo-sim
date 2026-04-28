"""Killer figure: bin-avg delta vs bin-avg phi vs k-truncation for zoom-in IC.

Three approaches to constructing the lo-res region of a zoom-in IC are
compared:
  A1: bin-average delta_{2N} -> delta_lo, then solve Poisson on coarse.
  A2: bin-average phi^{(1)}_{2N} -> phi_lo, then take gradient on coarse.
       Implied density delta_eff = -nabla^2 phi_lo differs from any
       physical density.
  B:  k-truncate the fine-grid white noise, convolve with T(k) -> clean
       band-limited delta_lo. Standard production-code approach.

Inside the zoom region (centered square), all three use delta_{2N} from
the fine grid. Outside, each uses its own lo-res construction. The
boundary discontinuity (or lack thereof) is the key visual.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- setup
rng = np.random.default_rng(42)
N, N2, L = 64, 128, 1.0
n_pk = -2.0
dx_f = L / N2
dx_c = L / N

# Fine k-grid.
kx = np.fft.fftfreq(N2, d=dx_f) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
K2_f = KX**2 + KY**2
T_f  = np.where(K2_f > 0, K2_f**(0.25*n_pk), 0.0)   # T(k) = sqrt(P(k)) = k^(n/2)
inv_K2_f = np.where(K2_f > 0, 1.0/K2_f, 0.0)

# Coarse k-grid.
kc = np.fft.fftfreq(N, d=dx_c) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')
K2_c = KXc**2 + KYc**2
inv_K2_c = np.where(K2_c > 0, 1.0/K2_c, 0.0)

# White noise -> fine-grid delta and phi.
mu = rng.standard_normal((N2, N2))
mu_k = np.fft.fftn(mu)
delta_k_2N = T_f * mu_k; delta_k_2N[0, 0] = 0.0
delta_2N = np.fft.ifftn(delta_k_2N).real
phi_k_2N = -delta_k_2N * inv_K2_f
phi_2N   = np.fft.ifftn(phi_k_2N).real

def restrict(field):
    """2x2 block average to coarse grid."""
    return field.reshape(N, 2, N, 2).mean(axis=(1, 3))

def laplacian(field, kxg, kyg):
    fk = np.fft.fftn(field)
    return np.fft.ifftn(-(kxg**2 + kyg**2) * fk).real

def gradient(field, kxg, kyg):
    fk = np.fft.fftn(field)
    return (np.fft.ifftn(1j*kxg*fk).real,
            np.fft.ifftn(1j*kyg*fk).real)

def poisson_solve(source, kxg, kyg):
    """Solve nabla^2 phi = source -> phi = -source/k^2 in Fourier."""
    K2g = kxg**2 + kyg**2
    inv = np.where(K2g > 0, 1.0/K2g, 0.0)
    return np.fft.ifftn(-np.fft.fftn(source) * inv).real

# ---------- Approach A1: bin-average delta ----------
delta_N_A1     = restrict(delta_2N)
phi_N_A1       = poisson_solve(delta_N_A1, KXc, KYc)
psix_A1, psiy_A1 = gradient(-phi_N_A1, KXc, KYc)
delta_implied_A1 = laplacian(phi_N_A1, KXc, KYc)

# ---------- Approach A2: bin-average phi ----------
phi_N_A2         = restrict(phi_2N)
psix_A2, psiy_A2 = gradient(-phi_N_A2, KXc, KYc)
delta_implied_A2 = laplacian(phi_N_A2, KXc, KYc)
# What someone might naively call the "lo-res density" in this approach:
delta_N_A2_naive = restrict(delta_2N)   # but this is NOT what Psi_lo describes!

# ---------- Approach B: k-truncate mu, convolve with T(k) ----------
mask        = (np.abs(KX) <= np.pi/dx_c) & (np.abs(KY) <= np.pi/dx_c)
mu_k_trunc  = mu_k * mask
delta_k_lp  = T_f * mu_k_trunc
delta_k_lp[0, 0] = 0.0
delta_lp_fine = np.fft.ifftn(delta_k_lp).real      # band-limited, on fine grid
delta_N_B     = delta_lp_fine[::2, ::2]            # exact sampling (band-limit)
phi_N_B       = poisson_solve(delta_N_B, KXc, KYc)
psix_B, psiy_B = gradient(-phi_N_B, KXc, KYc)
delta_implied_B = laplacian(phi_N_B, KXc, KYc)

# ---------- composites: hi-res inside zoom, lo-res outside ----------
def upsample_nn(coarse):
    return np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)

zi0, zi1 = N2//4, 3*N2//4
zoom_mask_fine = np.zeros((N2, N2), dtype=bool)
zoom_mask_fine[zi0:zi1, zi0:zi1] = True

def composite(d_in_fine, d_out_coarse):
    return np.where(zoom_mask_fine, d_in_fine, upsample_nn(d_out_coarse))

# For A1: outside-zoom density that the IC actually represents = delta_N_A1
# For A2: outside-zoom density that Psi describes = delta_implied_A2 (NOT delta_N_A2_naive)
# For B:  outside-zoom density = delta_N_B (= low-pass of delta_2N)
comp_A1 = composite(delta_2N, delta_N_A1)
comp_A2 = composite(delta_2N, delta_implied_A2)
comp_B  = composite(delta_2N, delta_N_B)

# ---------- residual against the IDEAL lowpass reference ----------
# delta_N_B is the clean k-truncated delta from the SAME mu, sampled on
# the coarse grid. By construction, sampling a band-limited field at
# coarse-grid points is exact (no aliasing). So delta_N_B is the right
# reference for what each approach's coarse-grid lo-res field should
# match.
# We compute residuals ON THE COARSE GRID to avoid upsampling artifacts:
res_coarse_A1 = delta_N_A1       - delta_N_B   # bin-avg minus ideal
res_coarse_A2 = delta_implied_A2 - delta_N_B   # what Psi_A2 implies, minus ideal
res_coarse_B  = delta_N_B        - delta_N_B   # exactly zero by construction
# (For A1 and A2 we also keep the fine-grid composite for the IC visualization.)

# Poisson-inconsistency residual on the coarse grid for each approach:
# poisson_resid = -nabla^2(phi_lo) - delta_lo_as-built.
# A1 and B are Poisson-consistent (zero); A2 is not.
poisson_resid_A1 = laplacian(phi_N_A1, KXc, KYc) - delta_N_A1
poisson_resid_A2 = laplacian(phi_N_A2, KXc, KYc) - delta_N_A2_naive
poisson_resid_B  = laplacian(phi_N_B,  KXc, KYc) - delta_N_B

# ---------------------------------------------------------------- figure
# 5 rows of diagnostics x 3 columns (one per approach).
fig, axes = plt.subplots(5, 3, figsize=(13, 14), constrained_layout=True,
                         gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.55]})
# Leave room on the left for the (unrotated) row labels; minimal padding.
fig.get_layout_engine().set(w_pad=2/72, h_pad=2/72, hspace=0.0, wspace=0.0,
                            rect=(0.07, 0.005, 0.995, 0.995))
approaches = [
    (r'A1: bin-avg $\delta$',         delta_N_A1,       res_coarse_A1, poisson_resid_A1, comp_A1),
    (r'A2: bin-avg $\phi^{(1)}$',     delta_implied_A2, res_coarse_A2, poisson_resid_A2, comp_A2),
    (r'B: k-trunc $\mu$ (standard)',  delta_N_B,        res_coarse_B,  poisson_resid_B,  comp_B),
]

vmax = float(np.max(np.abs(delta_2N)))
# Each diagnostic row has its own colour scale (they are different
# quantities). Within a row, the scale is shared across approaches so
# that the relative magnitudes are directly visible.
res_vmax = max(abs(res_coarse_A1).max(), abs(res_coarse_A2).max(), 1e-30)
poisson_vmax = max(abs(poisson_resid_A1).max(),
                   abs(poisson_resid_A2).max(),
                   abs(poisson_resid_B).max(), 1e-30)
x_coarse = (np.arange(N) + 0.5) * dx_c

row_labels = [
    'lo-res' + '\n' + r'$\delta_{\rm lo}$' + '\n' + '(as built)',
    'residual vs' + '\n' + 'ideal lowpass' + '\n' + r'$\delta_{\rm lo} - \delta_{\rm ideal\,lp}$',
    'Poisson' + '\n' + 'inconsistency' + '\n' + r'$-\nabla^2\phi_{\rm lo} - \delta_{\rm lo}$',
    'composite' + '\n' + 'IC density' + '\n' + '(hi-res inside)',
    'coarse-grid' + '\n' + 'slice' + '\n' + r'at $y = L/2$',
]

def fmt_rms(x):
    s = f'RMS = {x:.2e}'
    if x < 1e-12:
        s += '  (machine 0)'
    return s

for j, (name, dlo, res_coarse, pres, comp) in enumerate(approaches):
    # Row 0: lo-res delta itself.
    ax = axes[0, j]
    im = ax.imshow(dlo, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   origin='lower', extent=[0, L, 0, L])
    ax.add_patch(Rectangle((L/4, L/4), L/2, L/2, fill=False, edgecolor='lime', lw=2))
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(name)
    plt.colorbar(im, ax=ax, fraction=0.045)

    # Row 1: residual vs ideal lowpass.
    ax = axes[1, j]
    im = ax.imshow(res_coarse, cmap='RdBu_r', vmin=-res_vmax, vmax=res_vmax,
                   origin='lower', extent=[0, L, 0, L])
    ax.add_patch(Rectangle((L/4, L/4), L/2, L/2, fill=False, edgecolor='lime', lw=2))
    ax.set_xlabel('x'); ax.set_ylabel('y')
    rms = float(np.sqrt((res_coarse**2).mean()))
    ax.set_title(fmt_rms(rms))
    plt.colorbar(im, ax=ax, fraction=0.045)

    # Row 2: Poisson inconsistency map.
    ax = axes[2, j]
    im = ax.imshow(pres, cmap='RdBu_r', vmin=-poisson_vmax, vmax=poisson_vmax,
                   origin='lower', extent=[0, L, 0, L])
    ax.add_patch(Rectangle((L/4, L/4), L/2, L/2, fill=False, edgecolor='lime', lw=2))
    ax.set_xlabel('x'); ax.set_ylabel('y')
    rms = float(np.sqrt((pres**2).mean()))
    ax.set_title(fmt_rms(rms))
    plt.colorbar(im, ax=ax, fraction=0.045)

    # Row 3: composite IC density.
    ax = axes[3, j]
    im = ax.imshow(comp, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   origin='lower', extent=[0, L, 0, L])
    ax.add_patch(Rectangle((L/4, L/4), L/2, L/2, fill=False, edgecolor='lime', lw=2))
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.045)

    # Row 4: 1D coarse-grid slice through y = L/2.
    ax = axes[4, j]
    Iy0 = N // 2
    ax.plot(x_coarse, dlo[:, Iy0], 'C0o-', ms=4, lw=1.0,
            label=r'$\delta_{\rm lo}$')
    ax.plot(x_coarse, delta_N_B[:, Iy0], 'k--', lw=0.8,
            label=r'ideal lowpass')
    ax.axvline(L/4,   color='lime', lw=1.2, ls=':')
    ax.axvline(3*L/4, color='lime', lw=1.2, ls=':')
    ax.set_xlim(0, L)
    ax.set_xlabel('x'); ax.set_ylabel(r'$\delta_{\rm lo}$')
    ax.legend(fontsize=8)
    # Invisible spacer "colorbar" so the slice panel matches the image-panel width.
    cax = ax.inset_axes([1.02, 0, 0.04, 1.0])
    cax.axis('off')

# Row labels on the far left, aligned vertically.
for i, lbl in enumerate(row_labels):
    axes[i, 0].annotate(lbl, xy=(-0.40, 0.5), xycoords='axes fraction',
                        ha='center', va='center',
                        fontsize=12, fontweight='bold')

plt.savefig('zoom_ic_comparison.pdf')
print('wrote zoom_ic_comparison.pdf')

# ---------------------------------------------------------------- diagnostics
def relrms(a, b):
    return float(np.sqrt(((a-b)**2).mean()) / np.sqrt((a**2).mean()))

print('\n=== Quantitative diagnostics ===')
print('Poisson inconsistency RMS(delta_implied - delta_lo) / RMS(delta_lo):')
print(f"  A1 (bin-avg delta): {relrms(delta_implied_A1, delta_N_A1):.3e}  (Poisson-consistent)")
print(f"  A2 (bin-avg phi):   {relrms(delta_implied_A2, delta_N_A2_naive):.3e}  (NONZERO -> what Psi describes is NOT bin-avg delta)")
print(f"  B  (k-trunc mu):    {relrms(delta_implied_B,  delta_N_B):.3e}  (Poisson-consistent)")

# Boundary jump: RMS of delta along the boundary, comparing inside vs outside.
# Take the row of cells just inside vs just outside the zoom edge.
def boundary_jump(comp_field):
    edges = []
    # left edge of zoom: column zi0-1 (outside) vs zi0 (inside)
    edges.append(comp_field[zi0, zi0:zi1] - comp_field[zi0-1, zi0:zi1])
    edges.append(comp_field[zi1-1, zi0:zi1] - comp_field[zi1, zi0:zi1])
    edges.append(comp_field[zi0:zi1, zi0] - comp_field[zi0:zi1, zi0-1])
    edges.append(comp_field[zi0:zi1, zi1-1] - comp_field[zi0:zi1, zi1])
    j = np.concatenate(edges)
    return float(np.sqrt((j**2).mean()))

print('\nBoundary jump RMS(delta_inside - delta_outside) along zoom edge:')
print(f"  A1: {boundary_jump(comp_A1):.3e}")
print(f"  A2: {boundary_jump(comp_A2):.3e}")
print(f"  B:  {boundary_jump(comp_B):.3e}")

# For reference, the natural amplitude scale:
print(f"\nfor reference: RMS(delta_2N) = {float(np.sqrt((delta_2N**2).mean())):.3e}")
