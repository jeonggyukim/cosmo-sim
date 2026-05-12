"""3x4 diagnostic figure for the MUSIC2 matched-noise validation.

Test setup (Hahn 2011 §4.3, reproduced on the MUSIC2-anisotropic-zoom fork)
--------------------------------------------------------------------------
Two MUSIC2 runs are produced with bit-identical white noise on level
LEVELMAX inside a user-chosen patch:

    1. zoom:    levelmin < levelmax = LEVELMAX, ref_extent < 1, with the
                fork's options  kaveraging=no  +  density_boundary=yes  and a
                shared seed[LEVELMAX].  This generates noise on level LEVELMAX
                only inside the refined patch (size PS_xyz, lower corner
                PL_xyz, both reported by MUSIC2 in the music_*.log).
    2. unigrid: levelmin = levelmax = LEVELMAX over the full box (single-level
                grid of N = 2**LEVELMAX cells per side), same seed[LEVELMAX]
                integer.  This is the reference: what a full-box run at the
                patch resolution produces on its own.

Because kaveraging=no makes the per-cell noise coord-deterministic and
density_boundary=yes implements Hahn 2011's three-term δ assembly at the
coarse-fine boundary, the two grids should agree to ~1e-4 σ on δ(q) inside
the patch interior (away from the boundary).  This script measures and
visualises that residual.

What this figure shows (3 rows × 4 columns)
-------------------------------------------
Columns are δ(q) (Lagrangian density), Ψ_x, Ψ_y, Ψ_z (Lagrangian displacement
components), all read from the `generic`-format HDF5 dump
(`level_{LEVELMAX:03d}_DM_{rho,dx,dy,dz}`).

    Row 0 (zoom-in field):
        z-midplane of the zoom-in field itself, normalised by σ_unigrid.
        RdBu_r diverging colormap centred at zero.  Lets the viewer see the
        physical structure of δ/Ψ before looking at the residual.

    Row 1 (residual slice):
        z-midplane of  log10( |zoom - unigrid| / σ_unigrid ),  jet colormap,
        clipped to [-4, 0].  Title carries the patch-volume rms with margin=0
        (full patch) and margin=MARGIN (trim MARGIN cells from each face).
        The two rms values disagree when the boundary residual dominates.

    Row 2 (residual PDF):
        Log-x histogram of |Δq/σ_q| over (a) all patch cells (blue) and
        (b) the interior with margin=MARGIN (red).  Dashed verticals mark the
        two rms values.  When the boundary contaminates a tiny shell, the
        interior distribution shifts visibly to the left of the full-patch
        one.

Inputs (env vars)
-----------------
    LEVELMAX   level index of the refined grid (default 9 → N=512).
    MARGIN     number of cells trimmed from each face for the interior rms
               readout (default 16).
    TAG        run-family tag, e.g. 'matched_noise' or 'm2bc_2lpt'.
    LPT_TAG    LPT order tag in the run dirs, e.g. '1lpt' or '2lpt'.
    OUT        output PNG path; defaults to
               ~/Documents/music_validation/figures/matched_noise_{LPT_TAG}.png.

The script reads
    <base>/{TAG}_zoom_{LPT_TAG}/{TAG}_zoom_{LPT_TAG}_g.hdf5
    <base>/{TAG}_unigrid_{LPT_TAG}/{TAG}_unigrid_{LPT_TAG}_g.hdf5
and parses the zoom run's music_generic.log to recover the per-axis patch
size/lower-corner — so non-cubic (anisotropic) zoom patches work too.
"""
import os
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 11,
                     'axes.labelsize': 12, 'legend.fontsize': 10,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'figure.titlesize': 14})

# ---------------------------------------------------------------------------
# Geometry and run-identification, all from environment.
# ---------------------------------------------------------------------------
LEVELMAX = int(os.environ.get('LEVELMAX', '9'))
N = 1 << LEVELMAX                                   # unigrid resolution (cells per side)
MARGIN = int(os.environ.get('MARGIN', '16'))        # interior margin for rms readout
base = os.path.expanduser('~/Documents/music_validation')
TAG = os.environ.get('TAG', 'matched_noise')
LPT_TAG = os.environ.get('LPT_TAG', '1lpt')


def parse_zoom_placement(log_path):
    """Recover the level-LEVELMAX patch placement from MUSIC2's log.

    MUSIC2 prints the refinement hierarchy as
        Level 9   :   offset = (96, 96, 94)
                      size   = (104, 104, 105)
    where `offset` is given in *parent-level* cells (level LEVELMAX-1) and
    `size` is given directly in level-LEVELMAX cells.  We multiply offset by
    2 to put both quantities on the level-LEVELMAX grid, and return them as
        PL_xyz: (off_x, off_y, off_z)  — lower corner of the patch
        PS_xyz: (size_x, size_y, size_z) — patch extent
    both in level-LEVELMAX cells.  Supports anisotropic patches (different
    PS per axis).
    """
    txt = open(log_path).read()
    pattern = (rf'Level\s+{LEVELMAX}\s*:\s*offset\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
               rf'\s*\n[^\n]*size\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)')
    m = re.search(pattern, txt)
    if not m:
        raise RuntimeError(f'could not parse level-{LEVELMAX} placement from {log_path}')
    ox, oy, oz, sx, sy, sz = map(int, m.groups())
    return ((2*ox, 2*oy, 2*oz), (sx, sy, sz))


# We parse the zoom's *generic-format* log (music_generic.log).  The geometry
# is the same regardless of output format, but we settle on the generic log
# because that's the format whose HDF5 we actually load below.
zoom_log = f'{base}/{TAG}_zoom_{LPT_TAG}/music_generic.log'
PL_xyz, PS_xyz = parse_zoom_placement(zoom_log)
PS = max(PS_xyz)  # only used for status print; row-1 slice uses PS_xyz[2]
print(f'geometry: N={N}, PL_xyz={PL_xyz}, PS_xyz={PS_xyz}, MARGIN={MARGIN}')
zoom_stem = f'{TAG}_zoom_{LPT_TAG}'
uni_stem = f'{TAG}_unigrid_{LPT_TAG}'

# ---------------------------------------------------------------------------
# Load the four Lagrangian fields (density + 3 displacement components) from
# both runs.  Generic-format dumps store them under
#     level_NNN_DM_rho, level_NNN_DM_dx, level_NNN_DM_dy, level_NNN_DM_dz
# where NNN is the level index, zero-padded to three digits.
#
# Array shapes:
#   zoom dump (dz):  PS_xyz + 2*pad_zoom on each axis (the convolution
#                    padding that MUSIC2 carries around the refined patch).
#   unigrid dump (du): N + 2*pad_uni on each axis (the convolution padding
#                    around the top-level grid).
# Both are trimmed back to PS_xyz inside zp()/up() below.
# ---------------------------------------------------------------------------
with h5py.File(f'{base}/{zoom_stem}/{zoom_stem}_g.hdf5', 'r') as f:
    dz = [f[f'level_{LEVELMAX:03d}_DM_rho'][:], f[f'level_{LEVELMAX:03d}_DM_dx'][:],
          f[f'level_{LEVELMAX:03d}_DM_dy'][:], f[f'level_{LEVELMAX:03d}_DM_dz'][:]]
with h5py.File(f'{base}/{uni_stem}/{uni_stem}_g.hdf5', 'r') as f:
    du = [f[f'level_{LEVELMAX:03d}_DM_rho'][:], f[f'level_{LEVELMAX:03d}_DM_dx'][:],
          f[f'level_{LEVELMAX:03d}_DM_dy'][:], f[f'level_{LEVELMAX:03d}_DM_dz'][:]]


def zp(a):
    """Trim a zoom dump to the per-axis patch size.

    The zoom dump carries symmetric convolution padding around the patch
    (same number of cells on the +/- face per axis).  We compute the offset
    on each axis from the difference (shape - PS_xyz) // 2 and slice out the
    inner PS_xyz block.  Works for anisotropic patches because each axis is
    handled independently.
    """
    out = a
    for axis in range(3):
        s = a.shape[axis]
        ps = PS_xyz[axis]
        off = (s - ps) // 2
        out = out.take(np.arange(off, off + ps), axis=axis)
    return out


def up(a):
    """Trim a unigrid dump to the same patch in coords.

    The unigrid dump carries symmetric convolution padding (halo) around the
    full N**3 level-LEVELMAX grid.  We strip the halo first, then index into
    [PL_xyz[i], PL_xyz[i] + PS_xyz[i]) on each axis — the same patch
    location the zoom run carved out.  After this, zp(a_zoom) and up(a_uni)
    are PS_xyz-shaped arrays aligned cell-by-cell.
    """
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
mid = PS_xyz[2] // 2  # slice index along q_z for the 2D imshow panels

for col, (lbl, pz, pu) in enumerate(panels):
    # diff and sigma are computed on the full patch; rms_m is the same diff
    # restricted to the interior shell with MARGIN cells trimmed per face.
    diff = pz - pu
    sigma = pu.std()
    rms0 = np.sqrt((diff**2).mean()) / sigma
    rmsm = np.sqrt((diff[MARGIN:-MARGIN, MARGIN:-MARGIN, MARGIN:-MARGIN]**2).mean()) / sigma

    # ---- Row 0: the zoom-in field at q_z = mid, σ-normalised. ----
    # Diverging colormap (RdBu_r) centred at zero, with vmax = max(3σ, 99th
    # percentile of |slice|) so we always show at least ±3σ but expand for
    # rare large excursions.  Purely diagnostic — confirms the patch actually
    # contains physical structure of order unity.
    ax = axes[0, col]
    zoom_slice = pz[:, :, mid] / sigma
    vmax = max(3.0, np.percentile(np.abs(zoom_slice), 99))
    im = ax.imshow(zoom_slice.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f'{lbl}  zoom-in  (slice $q_z={mid}$)')
    ax.set_xlabel('$q_x$'); ax.set_ylabel('$q_y$')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label=r'$q_{\rm zoom}/\sigma_q$')

    # ---- Row 1: residual slice at q_z = mid. ----
    # log10(|Δq|/σ + 1e-12) on jet, clipped to [-4, 0].  The +1e-12 floor
    # avoids log(0) at any exactly-matching cell.  Title carries the m=0
    # (full patch) and m=MARGIN (interior) rms values: when both are similar,
    # the residual is uniform; when m=0 >> m=MARGIN, the boundary dominates.
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

    # ---- Row 2: log-x histogram of |Δq/σ|. ----
    # Two distributions overlaid: full patch (blue) and interior with
    # MARGIN-cell margin (red).  If the residual is concentrated at the
    # patch face, the red distribution sits visibly to the left of the blue
    # one and the two dashed rms verticals separate.  Bins are 71 edges
    # spanning log10 |Δq/σ| in [-6, 1] (i.e. residuals from 1e-6 σ to 10 σ).
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

# Figure-wide caption: identifies the MUSIC2 options under test, explains the
# three rows, and writes out the rms definition so a reader can interpret the
# title numbers without re-reading this script.
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
