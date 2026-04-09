"""
Plot a histogram of dr/dx, where dr is the displacement of each particle
from its nearest unperturbed lattice point.

Usage:
    python plot_dr_histogram.py [file1.hdf5 [file2.hdf5 ...]]

Defaults to comparing music_build/ics_swift.hdf5 (z=127) and ics_swift_z400.hdf5 (z=400).
"""

import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PTYPE = 1   # PartType1 = high-res DM in MUSIC2/SWIFT output

DEFAULT_FILES = [
    ('music_build/ics_swift.hdf5', 'z=127'),
    ('music_build/ics_swift_z400.hdf5', 'z=400'),
]


def compute_dr(fname):
    with h5py.File(fname, 'r') as f:
        hdr = dict(f['Header'].attrs)
        pos = f[f'PartType{PTYPE}/Coordinates'][()]

    box_size = hdr['BoxSize']
    if np.ndim(box_size) > 0:
        box_size = float(box_size[0])
    else:
        box_size = float(box_size)

    N_part = pos.shape[0]
    N_side = int(round(N_part ** (1.0 / 3.0)))
    assert N_side ** 3 == N_part, f"N_part={N_part} is not a perfect cube"

    dx = box_size / N_side

    # MUSIC2/SWIFT places unperturbed particles at (i + 0.5)*dx
    idx     = np.floor(pos / dx).astype(np.int64) % N_side
    lattice = (idx + 0.5) * dx
    delta   = pos - lattice
    delta  -= box_size * np.rint(delta / box_size)
    dr      = np.sqrt((delta ** 2).sum(axis=1))

    print(f"File      : {fname}")
    print(f"  N_part  : {N_part}  ({N_side}^3)")
    print(f"  Box     : {box_size:.4g} Mpc/h,  dx = {dx*1e3:.4g} kpc/h")
    print(f"  dr/dx   : min={dr.min()/dx:.4g}  max={dr.max()/dx:.4g}  mean={dr.mean()/dx:.4g}")

    return dr, delta, dx, box_size, N_side


# -----------------------------------------------------------------------
# Files to plot
# -----------------------------------------------------------------------
if len(sys.argv) > 1:
    files = [(f, f) for f in sys.argv[1:]]
else:
    files = DEFAULT_FILES

colors = ['steelblue', 'tomato', 'seagreen', 'darkorange']

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax_r, ax_comp = axes

comp_labels = [r'$d_x$', r'$d_y$', r'$d_z$']
comp_colors = ['C0', 'C1', 'C2']

for (fname, label), color in zip(files, colors):
    dr, delta, dx, box_size, N_side = compute_dr(fname)
    ax_r.hist(dr / dx, bins=200, color=color, edgecolor='none',
              alpha=0.6, label=label)
    for i, (cl, cc) in enumerate(zip(comp_labels, comp_colors)):
        ls = '-' if 'z=127' in label else '--'
        ax_comp.hist(delta[:, i] / dx, bins=200, color=cc, edgecolor='none',
                     alpha=0.4, histtype='stepfilled', linestyle=ls)

ax_r.axvline(1.0, color='black', ls='--', lw=1.0, label=r'$dr = dx$')
ax_r.set_xlabel(r'$dr\ /\ dx$', fontsize=13)
ax_r.set_ylabel('Count', fontsize=13)
ax_r.set_title('Displacement magnitude', fontsize=12)
ax_r.legend(fontsize=11)

# Component legend: color = component, linestyle = redshift
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(color=cc, alpha=0.7, label=cl) for cl, cc in zip(comp_labels, comp_colors)]
handles += [Line2D([0], [0], color='gray', ls='-',  label='z=127'),
            Line2D([0], [0], color='gray', ls='--', label='z=400')]
ax_comp.axvline(0, color='black', ls=':', lw=0.8)
ax_comp.set_xlabel(r'$d_i\ /\ dx$', fontsize=13)
ax_comp.set_ylabel('Count', fontsize=13)
ax_comp.set_title('Displacement components', fontsize=12)
ax_comp.legend(handles=handles, fontsize=10)

fig.suptitle(
    rf'{N_side}$^3$ particles, $L={box_size:.4g}$ Mpc/h',
    fontsize=12
)
plt.tight_layout()

plt.tight_layout()
outfile = 'dr_histogram.png'
plt.savefig(outfile, dpi=150)
print(f"\nSaved: {outfile}")
plt.show()
