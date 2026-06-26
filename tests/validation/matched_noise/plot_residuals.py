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
LEVELMIN = int(os.environ.get('LEVELMIN', str(LEVELMAX - 1)))
N = 1 << LEVELMAX                                   # unigrid resolution (cells per side)
N_COARSE = 1 << LEVELMIN                            # coarse-unigrid resolution
COARSE_STRIDE = 1 << (LEVELMAX - LEVELMIN)          # fine cells per coarse cell
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

# ---------------------------------------------------------------------------
# Exterior figure: comparison at COARSE resolution.
#
# Outside Ω the zoom run only stores a coarse-level field, so the
# comparison must be made at coarse resolution.  Apply the restriction
# operator (block-average 2^d → 1, i.e. the A1 kernel of
# restriction_lpt §3) to the fine-unigrid field; then compare cell-by-cell
# against the zoom's level-LEVELMIN field on the coarse grid.
#
# The restriction operator is the right coarse-grain for both δ
# (mass-weighted mean inside the block) and Ψ (centre-of-mass displacement
# of the 2^d sub-cells), so the same operator applies column-by-column.
#
#   Row 1: zoom coarse-level field (level LEVELMIN).
#   Row 2: restriction of fine-unigrid R[Ψ_fine] at coarse resolution
#          (same colour scale as row 1).
#   Row 3: residual zoom − R[fine-unigrid] at coarse resolution.
#          Inside the patch footprint the zoom intentionally couples to
#          the fine level; outside, the residual is the genuine
#          construction error of the zoom's coarse representation.
#   Row 4: residual pdf, outside Ω (green) vs inside Ω footprint (orange).
# ---------------------------------------------------------------------------
# Patch footprint at coarse resolution.
cPL = tuple(p // COARSE_STRIDE for p in PL_xyz)
cPS = tuple(s // COARSE_STRIDE for s in PS_xyz)
print(f'exterior geometry: N_COARSE={N_COARSE}, cPL={cPL}, cPS={cPS}')


def trim_halo(a, n):
    """Strip the symmetric convolution halo around an n**3 level grid."""
    out = a
    for axis in range(3):
        halo = (a.shape[axis] - n) // 2
        out = out.take(np.arange(halo, halo + n), axis=axis)
    return out


def block_avg(a, b):
    """Restrict an n**3 field to (n/b)**3 by averaging each b^3 block
    (the A1 kernel of restriction_lpt §3; cos-product window, attenuates
    coarse-Nyquist modes by ~3D corner factor ≈ 0.35 in amplitude)."""
    n = a.shape[0]
    nc = n // b
    return a.reshape(nc, b, nc, b, nc, b).mean(axis=(1, 3, 5))


def truncate_B(a, b):
    """Restrict an n**3 field to (n/b)**3 by sharp Fourier truncation
    inside the coarse Brillouin zone (the B kernel of restriction_lpt §3).
    No window attenuation up to coarse Nyquist; modes outside the coarse
    BZ are discarded.  Equivalent to projecting the fine field onto the
    subspace spanned by coarse-grid Fourier modes and evaluating at the
    coarse-grid positions."""
    n = a.shape[0]
    nc = n // b
    h = nc // 2
    F = np.fft.fftn(a)
    Fc = np.empty((nc, nc, nc), dtype=F.dtype)
    # Extract the 8 corner blocks of the fine FFT (low |k| modes per axis)
    # into the (nc, nc, nc) coarse FFT array.
    for sx, dx_dst, dx_src in ((0, slice(0, h),  slice(0, h)),
                               (1, slice(h, nc), slice(n - h, n))):
        for sy, dy_dst, dy_src in ((0, slice(0, h),  slice(0, h)),
                                   (1, slice(h, nc), slice(n - h, n))):
            for sz, dz_dst, dz_src in ((0, slice(0, h),  slice(0, h)),
                                       (1, slice(h, nc), slice(n - h, n))):
                Fc[dx_dst, dy_dst, dz_dst] = F[dx_src, dy_src, dz_src]
    # ifftn on coarse grid normalises by nc**3 instead of n**3; rescale so
    # the result equals the projection of the fine field onto coarse modes.
    return np.fft.ifftn(Fc).real * (nc / n) ** 3


# Zoom level-LEVELMIN field, already at coarse resolution.
with h5py.File(f'{base}/{zoom_stem}/{zoom_stem}_g.hdf5', 'r') as f:
    dz_c = [trim_halo(f[f'level_{LEVELMIN:03d}_DM_rho'][:], N_COARSE),
            trim_halo(f[f'level_{LEVELMIN:03d}_DM_dx'][:],  N_COARSE),
            trim_halo(f[f'level_{LEVELMIN:03d}_DM_dy'][:],  N_COARSE),
            trim_halo(f[f'level_{LEVELMIN:03d}_DM_dz'][:],  N_COARSE)]
# Restriction of fine-unigrid to coarse resolution.
#   du_c   : R_A1[Ψ_fine] = block average (cos-product window)
#   du_c_B : R_B [Ψ_fine] = sharp Fourier truncation inside coarse BZ
du_c   = [block_avg(trim_halo(u, N), COARSE_STRIDE) for u in du]
du_c_B = [truncate_B(trim_halo(u, N), COARSE_STRIDE) for u in du]


def upsample_nn(a, s):
    """Nearest-neighbour upsample by s along every axis."""
    out = a
    for ax in range(3):
        out = np.repeat(out, s, axis=ax)
    return out


# Fine-resolution display fields for rows 0 and 1 (rows 2+ stay at coarse
# resolution, since the residual analysis is at coarse).
#   zoom_disp_f: what the zoom IC actually carries — fine values inside Ω
#                and the coarse-level value broadcast (NN) to each 2^d
#                block of fine cells outside Ω.  Outside-Ω cells are
#                visibly blocky; inside-Ω cells carry the fine patch.
#   uni_disp_f : fine-unigrid truth at fine resolution.
zoom_fine_patches = [zp(a) for a in dz]
uni_disp_f = [trim_halo(u, N) for u in du]
zoom_disp_f = []
for zc, zf in zip(dz_c, zoom_fine_patches):
    full = upsample_nn(zc, COARSE_STRIDE)
    full[PL_xyz[0]:PL_xyz[0]+PS_xyz[0],
         PL_xyz[1]:PL_xyz[1]+PS_xyz[1],
         PL_xyz[2]:PL_xyz[2]+PS_xyz[2]] = zf
    zoom_disp_f.append(full)
mid_f = PL_xyz[2] + PS_xyz[2] // 2

# Boolean mask: True inside the patch footprint at coarse resolution.
ix = np.arange(N_COARSE)
inside_x = (ix >= cPL[0]) & (ix < cPL[0] + cPS[0])
inside_y = (ix >= cPL[1]) & (ix < cPL[1] + cPS[1])
inside_z = (ix >= cPL[2]) & (ix < cPL[2] + cPS[2])
mask_inside = inside_x[:, None, None] & inside_y[None, :, None] & inside_z[None, None, :]
mask_outside = ~mask_inside

# Slice index along q_z: patch z-midplane in coarse cells.
mid_c = cPL[2] + cPS[2] // 2

def shell_average(field_a, field_b=None, n_bins=None):
    """Spherical-shell average of <|A|^2> (if field_b is None) or
    Re<A B*> (cross-spectrum) over a 3D FFT grid.  Returns (k_bin_centres,
    average) with k in cell-index units (so k=1 means k = 2π/box-length on
    the coarse grid)."""
    n = field_a.shape[0]
    Fa = np.fft.fftn(field_a)
    if field_b is None:
        P = (Fa * Fa.conj()).real
    else:
        Fb = np.fft.fftn(field_b)
        P = (Fa * Fb.conj()).real
    kx = np.fft.fftfreq(n) * n
    K = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kx[None, None, :]**2)
    if n_bins is None:
        n_bins = n // 2
    bins = np.linspace(0, n // 2, n_bins + 1)
    idx = np.digitize(K.ravel(), bins) - 1
    Psum = np.zeros(n_bins); cnt = np.zeros(n_bins, dtype=int)
    mask = (idx >= 0) & (idx < n_bins)
    np.add.at(Psum, idx[mask], P.ravel()[mask])
    np.add.at(cnt,  idx[mask], 1)
    return 0.5 * (bins[:-1] + bins[1:]), Psum / np.maximum(cnt, 1)


fig2, axes2 = plt.subplots(6, 4, figsize=(20, 30))
for col, (lbl, pz, pu, pu_B) in enumerate(zip(labels, dz_c, du_c, du_c_B)):
    diff = pz - pu
    sigma = pu.std()
    rms_all = np.sqrt((diff**2).mean()) / sigma
    rms_out = np.sqrt((diff[mask_outside]**2).mean()) / sigma
    rms_in = np.sqrt((diff[mask_inside]**2).mean()) / sigma

    # Rows 0, 1 display at FINE resolution so the inside-Ω patch shows the
    # fine cells the zoom actually carries.  Outside Ω the zoom row is
    # nearest-upsampled coarse (visibly blocky); the unigrid row is fine
    # everywhere.  Normalise by each row's own σ so the colour scale spans
    # the field's natural range.
    zs_f = zoom_disp_f[col][:, :, mid_f]
    us_f = uni_disp_f[col][:, :, mid_f]
    sigma_zf = zoom_disp_f[col].std()
    sigma_uf = uni_disp_f[col].std()
    zslice = zs_f / sigma_zf
    uslice = us_f / sigma_uf
    vmax = max(3.0, np.percentile(np.abs(zslice), 99),
                    np.percentile(np.abs(uslice), 99))

    # Row 0: zoom IC at fine resolution.  Inside Ω: fine patch values;
    # outside Ω: each coarse value broadcast to its 2^d fine cells.
    ax = axes2[0, col]
    im = ax.imshow(zslice.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.add_patch(plt.Rectangle((PL_xyz[0] - 0.5, PL_xyz[1] - 0.5),
                               PS_xyz[0], PS_xyz[1],
                               fill=False, ec='k', lw=1.0, ls='--'))
    ax.set_title(f'{lbl}  zoom IC  (fine inside Ω, blocky coarse outside)')
    ax.set_xlabel('$q_x$ (fine cells)'); ax.set_ylabel('$q_y$ (fine cells)')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label=r'$q_{\rm zoom}/\sigma_{q,\rm zoom}$')

    # Row 1: fine-unigrid truth at fine resolution.
    ax = axes2[1, col]
    im = ax.imshow(uslice.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.add_patch(plt.Rectangle((PL_xyz[0] - 0.5, PL_xyz[1] - 0.5),
                               PS_xyz[0], PS_xyz[1],
                               fill=False, ec='k', lw=1.0, ls='--'))
    ax.set_title(f'{lbl}  fine-unigrid (truth at fine resolution)')
    ax.set_xlabel('$q_x$ (fine cells)'); ax.set_ylabel('$q_y$ (fine cells)')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label=r'$q_{\rm uni}/\sigma_{q,\rm uni}$')

    # Row 2: residual slice at coarse resolution.
    ax = axes2[2, col]
    lr = np.log10(np.abs(diff[:, :, mid_c]) / sigma + 1e-12)
    im = ax.imshow(lr.T, origin='lower', cmap='jet', vmin=-4, vmax=0)
    ax.add_patch(plt.Rectangle((cPL[0] - 0.5, cPL[1] - 0.5), cPS[0], cPS[1],
                               fill=False, ec='k', lw=1.0, ls='--'))
    ax.set_title(f'{lbl}  residual\n'
                 f'rms (all)       = {rms_all:.2e}\n'
                 f'rms (outside Ω) = {rms_out:.2e}\n'
                 f'rms (inside Ω)  = {rms_in:.2e}')
    ax.set_xlabel('$q_x$'); ax.set_ylabel('$q_y$')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label=r'$\log_{10}|\Delta q / \sigma_q|$')

    # Row 3: residual PDF, outside Ω vs inside Ω footprint, at coarse res.
    ax = axes2[3, col]
    out_vals = (diff[mask_outside] / sigma).ravel()
    in_vals = (diff[mask_inside] / sigma).ravel()
    bins = np.linspace(-6, 1, 71)
    ax.hist(np.log10(np.abs(out_vals) + 1e-12), bins=bins,
            color='C2', alpha=0.55,
            label=f'outside Ω  (N={out_vals.size:,})')
    ax.hist(np.log10(np.abs(in_vals) + 1e-12), bins=bins,
            color='C1', alpha=0.55,
            label=f'inside Ω footprint  (N={in_vals.size:,})')
    ax.axvline(np.log10(rms_out), color='C2', ls='--', lw=1)
    ax.axvline(np.log10(rms_in), color='C1', ls='--', lw=1)
    ax.set_xlim(-6, 1); ax.set_xlabel(r'$\log_{10}|\Delta q / \sigma_q|$')
    ax.set_ylabel('# of cells'); ax.set_yscale('log')
    ax.set_title(f'{lbl}  (residual distribution, coarse resolution)')
    ax.legend(loc='upper left', frameon=True, framealpha=0.85)

    # Row 4: Fourier-space diagnostics on the coarse grid.
    #   • amplitude ratios  sqrt(P_zoom(k)/P_R(k)) for R = A1 (solid) and B
    #     (dashed) restrictions of the fine field.  R_A1 attenuates near
    #     coarse Nyquist by the block-average window; R_B is flat inside
    #     the coarse BZ — direct comparison reveals which restriction is a
    #     fairer reference at high k.
    #   • cross-correlation  r(k) (against R_A1; against R_B is identical
    #     for k below the B truncation by construction).
    ax = axes2[4, col]
    k_bins, Pz_k   = shell_average(pz)
    _,       Pu_k   = shell_average(pu)
    _,       Pc_k   = shell_average(pz, pu)
    _,       Pu_kB  = shell_average(pu_B)
    amp_ratio_A1 = np.sqrt(np.maximum(Pz_k, 0) / np.maximum(Pu_k,  1e-300))
    amp_ratio_B  = np.sqrt(np.maximum(Pz_k, 0) / np.maximum(Pu_kB, 1e-300))
    xcorr = Pc_k / np.sqrt(np.maximum(Pz_k * Pu_k, 1e-300))
    ax.plot(k_bins, amp_ratio_A1, color='C0', lw=1.6,
            label=r'$\sqrt{P_{\rm zoom}/P_{R_{A1}[\mathrm{fine}]}}$')
    ax.plot(k_bins, amp_ratio_B,  color='C2', lw=1.6, ls='--',
            label=r'$\sqrt{P_{\rm zoom}/P_{R_{B}[\mathrm{fine}]}}$')
    ax.plot(k_bins, xcorr, color='C3', lw=1.2,
            label=r'$r(k)$ (vs $R_{A1}$)')
    ax.axhline(1, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('$k$  (cells$^{-1}$ on coarse grid)')
    ax.set_ylabel('ratio')
    ax.set_xscale('log')
    ax.set_xlim(1, N_COARSE // 2)
    ax.set_ylim(0, 1.4)
    ax.set_title(f'{lbl}  Fourier ratios')
    ax.legend(loc='lower left', fontsize=8, frameon=True, framealpha=0.85)

    # Row 5: residual slice using R_B[fine] (sharp Fourier truncation)
    # instead of R_A1 — exposes how much of the row-2 residual was the
    # block-average window vs the genuine noise mismatch.
    diff_B = pz - pu_B
    sigma_B = pu_B.std()
    rmsB_all = np.sqrt((diff_B**2).mean()) / sigma_B
    rmsB_out = np.sqrt((diff_B[mask_outside]**2).mean()) / sigma_B
    rmsB_in  = np.sqrt((diff_B[mask_inside ]**2).mean()) / sigma_B
    ax = axes2[5, col]
    lr = np.log10(np.abs(diff_B[:, :, mid_c]) / sigma_B + 1e-12)
    im = ax.imshow(lr.T, origin='lower', cmap='jet', vmin=-4, vmax=0)
    ax.add_patch(plt.Rectangle((cPL[0] - 0.5, cPL[1] - 0.5), cPS[0], cPS[1],
                               fill=False, ec='k', lw=1.0, ls='--'))
    ax.set_title(f'{lbl}  residual vs $R_B[\\mathrm{{fine}}]$\n'
                 f'rms (all)       = {rmsB_all:.2e}\n'
                 f'rms (outside Ω) = {rmsB_out:.2e}\n'
                 f'rms (inside Ω)  = {rmsB_in:.2e}')
    ax.set_xlabel('$q_x$'); ax.set_ylabel('$q_y$')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label=r'$\log_{10}|\Delta q / \sigma_q|$')

fig2.suptitle(
    'MUSIC2 fork: exterior matched-noise at COARSE resolution.\n'
    r'Reference is the restriction $R[\Psi_{\rm fine}]$ of the fine-unigrid; '
    r'$R_{A1}$ = $2^d\to 1$ block average, $R_B$ = sharp Fourier truncation '
    'inside coarse BZ.  Dashed rectangle = patch footprint.\n'
    'Row 1: zoom IC at fine resolution (fine inside Ω, blocky coarse '
    'outside).  '
    'Row 2: fine-unigrid (truth at fine resolution).  '
    r'Row 3: residual at COARSE resolution vs $R_{A1}[\mathrm{fine}]$.  '
    r'Row 4: residual pdf ($R_{A1}$).  '
    r'Row 5: Fourier amplitude ratios — $R_{A1}$ (solid) vs $R_B$ (dashed), '
    r'plus cross-correlation $r(k)$.  '
    r'Row 6: residual vs $R_B[\mathrm{fine}]$ at coarse resolution.')
fig2.tight_layout()
out_ext = out.replace('.png', '_exterior.png')
fig2.savefig(out_ext, dpi=300, bbox_inches='tight')
print(f'wrote {out_ext}')
