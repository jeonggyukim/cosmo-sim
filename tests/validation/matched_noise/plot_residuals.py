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

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 11,
                     'axes.labelsize': 12, 'legend.fontsize': 10,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'figure.titlesize': 14})

import re

LEVELMAX = int(os.environ.get('LEVELMAX', '9'))
N = 1 << LEVELMAX                                   # unigrid resolution
MARGIN = int(os.environ.get('MARGIN', '16'))        # interior margin for rms readout
base = os.path.expanduser('~/Documents/music_validation')
TAG = os.environ.get('TAG', 'matched_noise')
LPT_TAG = os.environ.get('LPT_TAG', '1lpt')


def parse_zoom_placement(log_path):
    """Read MUSIC2's music_*.log and recover the level-LEVELMAX patch
    placement.  MUSIC2 prints e.g.
        Level 9   :   offset = (96, 96, 94)
                      size   = (104, 104, 105)
    where offset and size are in level-(LEVELMAX-1) cells (the parent level).
    We multiply by 2 to convert to level-LEVELMAX cells.  Returns
    (PL_xyz, PS_xyz) in level-LEVELMAX cells.
    """
    txt = open(log_path).read()
    pattern = (rf'Level\s+{LEVELMAX}\s*:\s*offset\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
               rf'\s*\n[^\n]*size\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)')
    m = re.search(pattern, txt)
    if not m:
        raise RuntimeError(f'could not parse level-{LEVELMAX} placement from {log_path}')
    ox, oy, oz, sx, sy, sz = map(int, m.groups())
    # MUSIC2 reports `offset` in parent-level cells (level LEVELMAX-1) and
    # `size` directly in level-LEVELMAX cells.  Convert offset by ×2.
    return ((2*ox, 2*oy, 2*oz), (sx, sy, sz))


zoom_log = f'{base}/{TAG}_zoom_{LPT_TAG}/music_generic.log'
PL_xyz, PS_xyz = parse_zoom_placement(zoom_log)
PS = max(PS_xyz)  # for square panel; we crop down to per-axis below
print(f'geometry: N={N}, PL_xyz={PL_xyz}, PS_xyz={PS_xyz}, MARGIN={MARGIN}')
zoom_stem = f'{TAG}_zoom_{LPT_TAG}'
uni_stem = f'{TAG}_unigrid_{LPT_TAG}'

with h5py.File(f'{base}/{zoom_stem}/{zoom_stem}_g.hdf5', 'r') as f:
    dz = [f[f'level_{LEVELMAX:03d}_DM_rho'][:], f[f'level_{LEVELMAX:03d}_DM_dx'][:],
          f[f'level_{LEVELMAX:03d}_DM_dy'][:], f[f'level_{LEVELMAX:03d}_DM_dz'][:]]
with h5py.File(f'{base}/{uni_stem}/{uni_stem}_g.hdf5', 'r') as f:
    du = [f[f'level_{LEVELMAX:03d}_DM_rho'][:], f[f'level_{LEVELMAX:03d}_DM_dx'][:],
          f[f'level_{LEVELMAX:03d}_DM_dy'][:], f[f'level_{LEVELMAX:03d}_DM_dz'][:]]


def zp(a):
    """Trim zoom dump to the per-axis MUSIC2-reported patch size, using
    symmetric padding around the patch in each axis."""
    out = a
    for axis in range(3):
        s = a.shape[axis]
        ps = PS_xyz[axis]
        off = (s - ps) // 2
        out = out.take(np.arange(off, off + ps), axis=axis)
    return out


def up(a):
    """Strip unigrid halo, then take the per-axis patch [PL, PL+PS)."""
    out = a
    for axis in range(3):
        halo = (a.shape[axis] - N) // 2
        pl = PL_xyz[axis]
        ps = PS_xyz[axis]
        out = out.take(np.arange(halo + pl, halo + pl + ps), axis=axis)
    return out


labels = [r'$\delta(q)$', r'$\Psi_x$', r'$\Psi_y$', r'$\Psi_z$']
panels = [(lbl, zp(z), up(u)) for lbl, z, u in zip(labels, dz, du)]

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
mid = PS_xyz[2] // 2  # slice index along q_z

for col, (lbl, pz, pu) in enumerate(panels):
    diff = pz - pu
    sigma = pu.std()
    rms0 = np.sqrt((diff**2).mean()) / sigma
    rmsm = np.sqrt((diff[MARGIN:-MARGIN, MARGIN:-MARGIN, MARGIN:-MARGIN]**2).mean()) / sigma

    # Row 0: zoom-in's field itself at q_z = mid, sigma-normalised, diverging
    # colormap centred at zero.
    ax = axes[0, col]
    zoom_slice = pz[:, :, mid] / sigma
    vmax = max(3.0, np.percentile(np.abs(zoom_slice), 99))
    im = ax.imshow(zoom_slice.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f'{lbl}  zoom-in  (slice $q_z={mid}$)')
    ax.set_xlabel('$q_x$'); ax.set_ylabel('$q_y$')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label=r'$q_{\rm zoom}/\sigma_q$')

    # Row 1: residual slice — log10|Δq/σ_q|, jet colormap.
    ax = axes[1, col]
    lr = np.log10(np.abs(diff[:, :, mid]) / sigma + 1e-12)
    im = ax.imshow(lr.T, origin='lower', cmap='jet', vmin=-4, vmax=0)
    ax.set_title(f'{lbl}  residual\n'
                 f'rms (margin=0) = {rms0:.2e}\n'
                 f'rms (margin={MARGIN}) = {rmsm:.2e}')
    ax.set_xlabel('$q_x$'); ax.set_ylabel('$q_y$')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax,
                 label=r'$\log_{10}|\Delta q / \sigma_q|$')

    # Row 2: histogram of log10|Δq/σ_q| across all PS^3 particles, with the
    # 16-cell-margin interior overlaid as a separate distribution.
    ax = axes[2, col]
    all_vals = (diff / sigma).ravel()
    interior = (diff[MARGIN:-MARGIN, MARGIN:-MARGIN, MARGIN:-MARGIN] / sigma).ravel()
    bins = np.linspace(-6, 1, 71)
    ax.hist(np.log10(np.abs(all_vals) + 1e-12), bins=bins,
            color='C0', alpha=0.55,
            label=f'full patch  (N={all_vals.size:,})')
    ax.hist(np.log10(np.abs(interior) + 1e-12), bins=bins,
            color='C3', alpha=0.55,
            label=f'interior (margin={MARGIN}, N={interior.size:,})')
    ax.axvline(np.log10(rms0), color='C0', ls='--', lw=1)
    ax.axvline(np.log10(rmsm), color='C3', ls='--', lw=1)
    ax.set_xlim(-6, 1); ax.set_xlabel(r'$\log_{10}|\Delta q / \sigma_q|$')
    ax.set_ylabel('# of particles')
    ax.set_yscale('log')
    ax.set_title(f'{lbl}  (residual distribution)')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0),
              frameon=True, framealpha=0.85)
fig.suptitle(
    'MUSIC2 fork: kaveraging=no + density_boundary=yes\n'
    r'Row 1: zoom-in field at $q_z = PS/2$.  Row 2: residual slice.  '
    f'Row 3: residual pdf (full patch blue, interior margin={MARGIN} red).\n'
    r'rms $\equiv \langle |Q_{\rm zoom}(\mathbf{q}) - Q_{\rm uni}(\mathbf{q})|^2\rangle^{1/2}\, /\, \sigma_{Q,{\rm uni}}$,'
    r' averaged over Lagrangian cells (margin = number of cells trimmed from each face).')
fig.tight_layout()
out = os.environ.get('OUT', f'{base}/figures/matched_noise_{LPT_TAG}.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'wrote {out}')
