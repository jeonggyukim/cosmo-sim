#!/usr/bin/env python3
"""
plot_mpi_fft_slabs.py — how a distributed 3D FFT is organised, and what
"transposed output" means for the index order in Fourier space.

A 3D FFT is three sets of 1D transforms, one per axis. Under MPI the array is
cut into slabs along one axis, so two of the three directions lie inside a slab
and can be transformed without any communication; the third cannot, and needs a
global redistribution. FFTW can stop after that single redistribution and hand
back an array whose first two axes are swapped, saving a second all-to-all.
monofonIC uses that option (mpi_local_size_3d_transposed in src/grid_fft.cc),
which is why its k-space array is indexed [k_y][k_x][k_z].
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

N = 8
NRANK = 4
PER = N // NRANK
COLS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def slab_grid(ax, axis_labels, slab_axis, title, hatchlast=False):
    """Draw an N x N array coloured by which rank owns which slab.

    slab_axis = 0 slices rows (the vertical axis), 1 slices columns.
    """
    for r in range(NRANK):
        for m in range(PER):
            idx = r * PER + m
            if slab_axis == 0:
                rect = Rectangle((0, idx), N, 1, fc=COLS[r], ec="w", lw=0.6, alpha=0.85)
            else:
                rect = Rectangle((idx, 0), 1, N, fc=COLS[r], ec="w", lw=0.6, alpha=0.85)
            ax.add_patch(rect)
    for r in range(NRANK):
        if slab_axis == 0:
            ax.text(N + 0.35, r * PER + PER / 2, f"rank {r}", va="center",
                    fontsize=7.5, color=COLS[r])
        else:
            ax.text(r * PER + PER / 2, N + 0.35, f"rank {r}", ha="center",
                    fontsize=7.5, color=COLS[r], rotation=0)
    ax.set_xlim(-0.4, N + 2.6 if slab_axis == 0 else N + 0.4)
    ax.set_ylim(-4.4, N + 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(N / 2, -0.75, axis_labels[1], ha="center", va="top", fontsize=10)
    ax.set_ylabel(axis_labels[0], fontsize=10)
    ax.set_title(title, fontsize=9.5)


fig = plt.figure(figsize=(12.0, 7.4))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30)

# (a) real space, sliced along x
ax = fig.add_subplot(gs[0, 0])
slab_grid(ax, [r"$x$  (sliced)", r"$y$"], 0,
          r"(a)  real space: slabs along $x$")
ax.text(N / 2, -2.4, r"$z$ runs into the page; each rank holds"
                     "\n" r"$N/4$ values of $x$, all $y$ and all $z$",
        ha="center", va="top", fontsize=7.5, color="0.35")

# (b) transforms along the two local axes
ax = fig.add_subplot(gs[0, 1])
slab_grid(ax, [r"$x$  (sliced)", r"$k_y$"], 0,
          r"(b)  FFT along $y$ and $z$: no communication")
ax.text(N / 2, -2.4, "both axes lie inside a slab, so each\n"
                     "rank transforms its own data alone",
        ha="center", va="top", fontsize=7.5, color="0.35")

# (c) the all-to-all
ax = fig.add_subplot(gs[0, 2])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title(r"(c)  transposing $x \leftrightarrow k_y$: one all-to-all", fontsize=9.5)
for r in range(NRANK):
    ax.add_patch(Rectangle((0.04, 0.60 - 0.11 * r), 0.28, 0.09,
                           fc=COLS[r], alpha=0.85, ec="w"))
    ax.add_patch(Rectangle((0.68 + 0.07 * r, 0.24), 0.06, 0.45,
                           fc=COLS[r], alpha=0.85, ec="w"))
for r in range(NRANK):
    ax.add_patch(FancyArrowPatch((0.34, 0.645 - 0.11 * r),
                                 (0.66 + 0.07 * r, 0.70),
                                 arrowstyle="-|>", mutation_scale=8,
                                 color=COLS[r], lw=0.9, alpha=0.8,
                                 connectionstyle="arc3,rad=0.18"))
ax.text(0.18, 0.74, "slabs of $x$", ha="center", fontsize=8)
ax.text(0.81, 0.16, "slabs of $k_y$", ha="center", fontsize=8)
ax.text(0.0, 0.08,
        "To transform along $x$ every rank needs all $x$\n"
        "for the modes it will keep, so the array is\n"
        "redistributed once. This is the only\n"
        "communication in the whole transform.",
        fontsize=7.5, va="top", color="0.35")

# (d) k space, transposed
ax = fig.add_subplot(gs[1, 0])
slab_grid(ax, [r"$k_y$  (sliced)", r"$k_x$"], 0,
          r"(d)  Fourier space, transposed output")
ax.text(N / 2, -2.4, "the first array index is now $k_y$;\n"
                     "FFTW skips a second all-to-all",
        ha="center", va="top", fontsize=7.5, color="0.35")

# (e) the index swap for one mode
ax = fig.add_subplot(gs[1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title(r"(e)  one mode, two layouts", fontsize=9.5)
ax.text(0.5, 0.86, r"physical mode  $(n_x, n_y, k_z) = (2, 5, k)$",
        ha="center", fontsize=9)
rows = [("serial", r"array$[2][5][k]$", r"SeedTable$[\,i\,N + j\,]$", "0.25"),
        ("MPI, transposed", r"array$[5][2][k]$", r"SeedTable$[\,j\,N + i\,]$", "#c44e52")]
ax.text(0.02, 0.70, "layout", fontsize=7.5, color="0.45")
ax.text(0.42, 0.70, "array index", fontsize=7.5, color="0.45")
ax.text(0.74, 0.70, "seed lookup", fontsize=7.5, color="0.45")
ax.plot([0.0, 1.0], [0.66, 0.66], color="0.85", lw=0.8)
for m, (lab, arr, seed, col) in enumerate(rows):
    y = 0.55 - 0.16 * m
    ax.text(0.02, y, lab, fontsize=8, color=col)
    ax.text(0.42, y, arr, fontsize=8.5, color=col)
    ax.text(0.74, y, seed, fontsize=8.5, color=col)
ax.text(0.0, 0.28,
        "The lookup swaps $i$ and $j$ in the distributed\n"
        "branch precisely so the same physical mode keeps\n"
        "the same seed under either layout.",
        fontsize=7.5, va="top", color="0.35")

# (f) why it matters
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.set_title("(f)  why the field is decomposition-independent", fontsize=9.5, loc="left")
ax.text(0, 0.94,
        "Each rank runs the whole loop over $(n_x, n_y)$ and\n"
        "redraws every random number, writing only the\n"
        "slabs it owns. No rank ever needs a number that\n"
        "another rank generated, so the transform above is\n"
        "the only communication in the generator.\n\n"
        "A mode and its conjugate partner can land on\n"
        "different ranks, which is why both are tested\n"
        "before writing.\n\n"
        "The realisation therefore depends only on the\n"
        r"map $(n_x,n_y) \rightarrow$ seed, never on how the work"
        "\n"
        "was divided: 1 rank and 64 ranks give the same\n"
        "field to the last bit.",
        fontsize=8, va="top", linespacing=1.55)

fig.suptitle("Slab-decomposed FFT and what \"transposed output\" means",
             fontsize=12, y=0.98)
fig.savefig("mpi_fft_slabs.pdf", bbox_inches="tight")
fig.savefig("mpi_fft_slabs.png", dpi=150, bbox_inches="tight")
print("Saved: mpi_fft_slabs.pdf, mpi_fft_slabs.png")
