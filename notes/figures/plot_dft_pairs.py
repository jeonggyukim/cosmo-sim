#!/usr/bin/env python3
"""
plot_dft_pairs.py — Four illustrative DFT input/output pairs for fft_review.tex.

Panels (left = f_n, right = |f̃_k|):
  (a) Unit impulse at n=0       → flat spectrum
  (b) Constant signal           → DC spike only
  (c) Cosine (k0=3 out of N=16) → two symmetric spikes
  (d) Gaussian                  → narrower Gaussian in k-space
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

N = 16
n = np.arange(N)
k = np.arange(N)

# --- signals and their DFTs ---
k0 = 3   # cosine frequency index
sigma = 2.5  # Gaussian width (samples)

signals = {
    r"(a) $f_n = \delta_{n,0}$ (unit impulse)",
    r"(b) $f_n = 1$ (constant)",
    r"(c) $f_n = \cos(2\pi \cdot 3 \cdot n/N)$",
    r"(d) Gaussian $e^{-n^2/(2\sigma^2)}$",
}

f_list = [
    np.array([1.0] + [0.0]*(N-1)),
    np.ones(N),
    np.cos(2*np.pi*k0*n/N),
    np.exp(-((n - N//2)**2)/(2*sigma**2)),   # centred Gaussian
]
labels = [
    r"(a) $f_n = \delta_{n,0}$",
    r"(b) $f_n = 1$",
    r"(c) $f_n = \cos(2\pi\cdot 3\cdot n/N)$",
    r"(d) Gaussian ($\sigma=2.5$)",
]
spec_labels = [
    r"$|\tilde{f}_k| = 1$",
    r"$|\tilde{f}_k| = N\delta_{k,0}$",
    r"$|\tilde{f}_k| = (N/2)(\delta_{k,3}+\delta_{k,13})$",
    r"Gaussian in $k$",
]

fig = plt.figure(figsize=(9, 7))
gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.38,
              left=0.09, right=0.97, top=0.96, bottom=0.06)

colours = ["#2166ac", "#d6604d", "#1a9850", "#762a83"]

for i, (f, lab, slab) in enumerate(zip(f_list, labels, spec_labels)):
    F = np.fft.fft(f)
    absF = np.abs(F)

    ax_l = fig.add_subplot(gs[i, 0])
    ax_r = fig.add_subplot(gs[i, 1])

    ax_l.stem(n, f, linefmt=colours[i], markerfmt='o',
              basefmt='k-', )
    ax_l.set_xlim(-0.5, N-0.5)
    ax_l.set_xlabel(r"$n$", fontsize=8)
    ax_l.set_ylabel(r"$f_n$", fontsize=8)
    ax_l.set_title(lab, fontsize=8, loc='left')
    ax_l.tick_params(labelsize=7)
    ax_l.axhline(0, color='k', lw=0.5)

    ax_r.stem(k, absF, linefmt=colours[i], markerfmt='o',
              basefmt='k-', )
    ax_r.set_xlim(-0.5, N-0.5)
    ax_r.set_xlabel(r"$k$", fontsize=8)
    ax_r.set_ylabel(r"$|\tilde{f}_k|$", fontsize=8)
    ax_r.set_title(slab, fontsize=8, loc='left')
    ax_r.tick_params(labelsize=7)
    ax_r.axhline(0, color='k', lw=0.5)

# column headers
fig.text(0.27, 0.985, "Real space  $f_n$",
         ha='center', va='top', fontsize=9, fontweight='bold')
fig.text(0.73, 0.985, "Frequency space  $|\\tilde{f}_k|$",
         ha='center', va='top', fontsize=9, fontweight='bold')

fig.savefig("dft_pairs.pdf", bbox_inches="tight")
fig.savefig("dft_pairs.png", dpi=150, bbox_inches="tight")
print("Saved: dft_pairs.pdf, dft_pairs.png")
