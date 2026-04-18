#!/usr/bin/env python3
"""
plot_dft_pairs.py — Four illustrative DFT input/output pairs for fft.tex.

Panels (left = f_n, right = |f̃_k| with fftshift so k=0 is centred):
  (a) Unit impulse at n=0       → flat spectrum
  (b) Constant signal           → DC spike only
  (c) Cosine (k0=3 out of N=16) → two symmetric spikes at ±k0
  (d) Gaussian (σ=1.5)          → broader Gaussian in k-space (uncertainty principle)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

N = 16
n = np.arange(N)
# Shifted frequency axis: -N/2 … N/2-1
k_shift = np.fft.fftshift(np.arange(N) - N // 2)   # [-8,-7,...,7]
k_shift = np.arange(-N // 2, N // 2)                # cleaner: -8..7

k0    = 3      # cosine frequency index
sigma = 1.5    # Gaussian width (samples) — narrow in x → broad in k

f_list = [
    np.array([1.0] + [0.0] * (N - 1)),               # (a) impulse at n=0
    np.ones(N),                                        # (b) constant
    np.cos(2 * np.pi * k0 * n / N),                   # (c) cosine at k0=3
    np.exp(-n**2 / (2 * sigma**2)),                    # (d) Gaussian at n=0 (wraps)
]
labels = [
    r"(a) $f_n = \delta_{n,0}$",
    r"(b) $f_n = 1$",
    r"(c) $f_n = \cos(2\pi\cdot 3\cdot n/N)$",
    r"(d) Gaussian at $n=0$, $\sigma=1.5$",
]
spec_labels = [
    r"$|\tilde{f}_k| = 1$  (flat)",
    r"$|\tilde{f}_k| = N\delta_{k,0}$  (DC only)",
    r"$|\tilde{f}_k|$: spikes at $k=\pm 3$",
    r"Gaussian in $k$  (broader, $\sigma_k \approx 1.7$)",
]

colours = ["#2166ac", "#d6604d", "#1a9850", "#762a83"]

fig = plt.figure(figsize=(9, 7.5))
gs = GridSpec(4, 2, figure=fig, hspace=0.65, wspace=0.38,
              left=0.09, right=0.97, top=0.91, bottom=0.06)

for i, (f, lab, slab) in enumerate(zip(f_list, labels, spec_labels)):
    F     = np.fft.fft(f)
    absF  = np.fft.fftshift(np.abs(F))   # centre at k=0

    ax_l = fig.add_subplot(gs[i, 0])
    ax_r = fig.add_subplot(gs[i, 1])

    # Real-space panel
    ax_l.stem(n, f, linefmt=colours[i], markerfmt='o', basefmt='k-')
    ax_l.set_xlim(-0.5, N - 0.5)
    ax_l.set_xlabel(r"$n$", fontsize=8)
    ax_l.set_ylabel(r"$f_n$", fontsize=8)
    ax_l.set_title(lab, fontsize=8, loc='left')
    ax_l.tick_params(labelsize=7)
    ax_l.axhline(0, color='k', lw=0.5)

    # Frequency-space panel (fftshifted, k centred at 0)
    ax_r.stem(k_shift, absF, linefmt=colours[i], markerfmt='o', basefmt='k-')
    ax_r.set_xlim(-N // 2 - 0.5, N // 2 - 0.5)
    ax_r.set_xlabel(r"$k$", fontsize=8)
    ax_r.set_ylabel(r"$|\tilde{f}_k|$", fontsize=8)
    ax_r.set_title(slab, fontsize=8, loc='left')
    ax_r.tick_params(labelsize=7)
    ax_r.axhline(0, color='k', lw=0.5)

# Column headers — above the grid with enough clearance
fig.text(0.27, 0.975, "Real space  $f_n$",
         ha='center', va='top', fontsize=10, fontweight='bold')
fig.text(0.73, 0.975, "Frequency space  $|\\tilde{f}_k|$",
         ha='center', va='top', fontsize=10, fontweight='bold')

fig.savefig("dft_pairs.pdf", bbox_inches="tight")
fig.savefig("dft_pairs.png", dpi=150, bbox_inches="tight")
print("Saved: dft_pairs.pdf, dft_pairs.png")
