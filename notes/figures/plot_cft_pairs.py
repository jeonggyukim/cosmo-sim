#!/usr/bin/env python3
"""
plot_cft_pairs.py — Four illustrative continuous FT pairs for fft_review.tex.

Panels (left = f(x), right = |f̂(ξ)|):
  (a) Delta function at x0        → flat spectrum
  (b) Gaussian (narrow σ)         → broad Gaussian in ξ (uncertainty principle)
  (c) Top-hat / rect of width L   → sinc (ringing)
  (d) Cosine at frequency ξ0      → two symmetric spikes

Convention: f̂(ξ) = ∫ f(x) e^{-2πiξx} dx
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Common x and xi grids
x  = np.linspace(-6, 6, 4000)
xi = np.linspace(-3, 3, 4000)
dx = x[1] - x[0]


def cft(f, x, xi):
    """Numerical continuous FT via trapezoidal rule."""
    F = np.array([np.trapezoid(f * np.exp(-2j * np.pi * xiv * x), x) for xiv in xi])
    return F


# ---- (a) delta function approximated by narrow Gaussian ----
sigma_d = 0.15
f_a = np.exp(-np.pi * x**2 / sigma_d**2)
f_a /= np.trapezoid(f_a, x)          # normalise to unit area
F_a = np.abs(cft(f_a, x, xi))

# ---- (b) Gaussian: narrow in x → broad in ξ ----
sigma_b = 0.7
f_b = np.exp(-np.pi * x**2 / sigma_b**2)
F_b = np.abs(cft(f_b, x, xi))    # should be σ_b * exp(-π σ_b^2 ξ^2)

# ---- (c) Top-hat of width L ----
L = 2.0
f_c = np.where(np.abs(x) <= L/2, 1.0, 0.0)
F_c = np.abs(cft(f_c, x, xi))    # should be L sinc(L ξ)

# ---- (d) Cosine at ξ0 ----
xi0 = 1.0
f_d = np.cos(2 * np.pi * xi0 * x) * np.exp(-x**2 / 8)  # windowed for display
F_d = np.abs(cft(f_d, x, xi))

datasets = [
    (f_a, F_a,
     r"(a) $f(x) \approx \delta(x)$  (narrow Gaussian)",
     r"$|\hat{f}(\xi)| \approx 1$  (flat)"),
    (f_b, F_b,
     r"(b) $f(x) = e^{-\pi x^2/\sigma^2}$,  $\sigma=0.7$",
     r"$|\hat{f}(\xi)| = \sigma\,e^{-\pi\sigma^2\xi^2}$  (broader Gaussian)"),
    (f_c, F_c,
     r"(c) $f(x) = \mathrm{rect}(x/L)$,  $L=2$",
     r"$|\hat{f}(\xi)| = L|\mathrm{sinc}(L\xi)|$"),
    (f_d, F_d,
     r"(d) $f(x) = \cos(2\pi\xi_0 x)$ (windowed), $\xi_0=1$",
     r"$|\hat{f}(\xi)|$: two spikes at $\pm\xi_0$"),
]

colours = ["#2166ac", "#d6604d", "#1a9850", "#762a83"]

fig = plt.figure(figsize=(9, 7))
gs = GridSpec(4, 2, figure=fig, hspace=0.6, wspace=0.38,
              left=0.09, right=0.97, top=0.96, bottom=0.06)

for i, (f, F, lab, flab) in enumerate(datasets):
    ax_l = fig.add_subplot(gs[i, 0])
    ax_r = fig.add_subplot(gs[i, 1])

    ax_l.plot(x, f, color=colours[i], lw=1.3)
    ax_l.set_xlim(-5, 5)
    ax_l.set_xlabel(r"$x$", fontsize=8)
    ax_l.set_ylabel(r"$f(x)$", fontsize=8)
    ax_l.set_title(lab, fontsize=8, loc='left')
    ax_l.axhline(0, color='k', lw=0.5)
    ax_l.tick_params(labelsize=7)

    ax_r.plot(xi, F, color=colours[i], lw=1.3)
    ax_r.set_xlim(-3, 3)
    ax_r.set_xlabel(r"$\xi$", fontsize=8)
    ax_r.set_ylabel(r"$|\hat{f}(\xi)|$", fontsize=8)
    ax_r.set_title(flab, fontsize=8, loc='left')
    ax_r.axhline(0, color='k', lw=0.5)
    ax_r.tick_params(labelsize=7)

fig.text(0.27, 0.985, "Real space  $f(x)$",
         ha='center', va='top', fontsize=9, fontweight='bold')
fig.text(0.73, 0.985, "Frequency space  $|\\hat{f}(\\xi)|$",
         ha='center', va='top', fontsize=9, fontweight='bold')

fig.savefig("cft_pairs.pdf", bbox_inches="tight")
fig.savefig("cft_pairs.png", dpi=150, bbox_inches="tight")
print("Saved: cft_pairs.pdf, cft_pairs.png")
