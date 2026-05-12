"""1x4 panel for MUSIC2 fork with kaveraging=no + density_boundary=yes.

Hahn 2011 §4.3 matched-noise validation on the Lagrangian convolution grid:
δ(q), Ψ_x, Ψ_y, Ψ_z from the generic-format dump of both zoom and unigrid.

Each panel shows the z-midplane of log10(|zoom - unigrid|/σ_unigrid) at the
patch interior, paired by Lagrangian cell.  rms (m=0) and rms (m=16) reported
in the title.
"""
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

PS = 104   # patch size in level-9 cells (ref_extent=0.2 × 512)
PL = 204   # patch lower corner in level-9 cells
N = 512    # unigrid level-max resolution
base = os.path.expanduser('~/Documents/music_validation')
TAG = os.environ.get('TAG', 'matched_noise')
LPT_TAG = os.environ.get('LPT_TAG', '1lpt')
zoom_stem = f'{TAG}_zoom_{LPT_TAG}'
uni_stem = f'{TAG}_unigrid_{LPT_TAG}'

with h5py.File(f'{base}/{zoom_stem}/{zoom_stem}_g.hdf5', 'r') as f:
    dz = [f['level_009_DM_rho'][:], f['level_009_DM_dx'][:],
          f['level_009_DM_dy'][:], f['level_009_DM_dz'][:]]
with h5py.File(f'{base}/{uni_stem}/{uni_stem}_g.hdf5', 'r') as f:
    du = [f['level_009_DM_rho'][:], f['level_009_DM_dx'][:],
          f['level_009_DM_dy'][:], f['level_009_DM_dz'][:]]


def zp(a):
    off = (a.shape[0] - PS) // 2
    return a[off:off+PS, off:off+PS, off:off+PS]


def up(a):
    halo = (a.shape[0] - N) // 2
    return a[halo+PL:halo+PL+PS, halo+PL:halo+PL+PS, halo+PL:halo+PL+PS]


labels = [r'$\delta(q)$', r'$\Psi_x$', r'$\Psi_y$', r'$\Psi_z$']
panels = [(lbl, zp(z), up(u)) for lbl, z, u in zip(labels, dz, du)]

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
mid = PS // 2

for col, (lbl, pz, pu) in enumerate(panels):
    diff = pz - pu
    sigma = pu.std()
    rms0 = np.sqrt((diff**2).mean()) / sigma
    rmsm = np.sqrt((diff[16:-16, 16:-16, 16:-16]**2).mean()) / sigma

    # Row 0: zoom-in's field itself at q_z = mid, sigma-normalised, diverging
    # colormap centred at zero.
    ax = axes[0, col]
    zoom_slice = pz[:, :, mid] / sigma
    vmax = max(3.0, np.percentile(np.abs(zoom_slice), 99))
    im = ax.imshow(zoom_slice.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f'{lbl}  zoom-in  (slice $q_z={mid}$)', fontsize=10)
    ax.set_xlabel('$q_x$'); ax.set_ylabel('$q_y$')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label=r'$q_{\rm zoom}/\sigma_q$')

    # Row 1: residual slice — log10|Δq/σ_q|, jet colormap.
    ax = axes[1, col]
    lr = np.log10(np.abs(diff[:, :, mid]) / sigma + 1e-12)
    im = ax.imshow(lr.T, origin='lower', cmap='jet', vmin=-4, vmax=0)
    ax.set_title(f'{lbl}  residual  (slice $q_z={mid}$)\n'
                 f'rms (margin=0) = {rms0:.2e}, '
                 f'rms (margin=16) = {rmsm:.2e}',
                 fontsize=10)
    ax.set_xlabel('$q_x$'); ax.set_ylabel('$q_y$')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax,
                 label=r'$\log_{10}|\Delta q / \sigma_q|$')

    # Row 2: histogram of log10|Δq/σ_q| across all PS^3 particles, with the
    # 16-cell-margin interior overlaid as a separate distribution.
    ax = axes[2, col]
    all_vals = (diff / sigma).ravel()
    interior = (diff[16:-16, 16:-16, 16:-16] / sigma).ravel()
    bins = np.linspace(-6, 1, 71)
    ax.hist(np.log10(np.abs(all_vals) + 1e-12), bins=bins,
            color='C0', alpha=0.55,
            label=f'full patch  (N={all_vals.size:,})')
    ax.hist(np.log10(np.abs(interior) + 1e-12), bins=bins,
            color='C3', alpha=0.55,
            label=f'16-cell margin (N={interior.size:,})')
    ax.axvline(np.log10(rms0), color='C0', ls='--', lw=1)
    ax.axvline(np.log10(rmsm), color='C3', ls='--', lw=1)
    ax.set_xlim(-6, 1); ax.set_xlabel(r'$\log_{10}|\Delta q / \sigma_q|$')
    ax.set_ylabel('N particles')
    ax.set_yscale('log')
    ax.set_title(f'{lbl}  (residual distribution)', fontsize=10)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
fig.suptitle(
    'MUSIC2 fork: kaveraging=no + density_boundary=yes (Hahn 2011 §4.3 setup)\n'
    r'Row 1: zoom-in field at $q_z = PS/2$.  Row 2: residual slice.  '
    r'Row 3: residual pdf (full patch blue, 16-cell margin interior red).',
    fontsize=12)
fig.tight_layout()
out = os.environ.get('OUT', f'{base}/figures/matched_noise_{LPT_TAG}.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'wrote {out}')
