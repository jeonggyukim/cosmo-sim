#!/usr/bin/env python3
"""
plot_cft_pairs.py — Four illustrative continuous FT pairs for fft.tex.

Convention B (physics/cosmology):
  f̂(k) = ∫ f(x) e^{-ikx} dx,   f(x) = ∫ dk/(2π) f̂(k) e^{ikx}

Panels (left = f(x), right = |f̂(k)|):
  (a) Narrow Gaussian (≈ δ)      → flat spectrum |f̂(k)| ≈ 1
  (b) Gaussian σ=0.7             → broader Gaussian (uncertainty principle)
  (c) Top-hat width L=2          → |2sin(kL/2)/k|  (sinc-like, ringing)
  (d) Windowed cosine at k₀=1.5  → two symmetric spikes at ±k₀
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

# Real-space grid
x = np.linspace(-6, 6, 4000)
dx = x[1] - x[0]

# Wavenumber grid (Convention B: k in rad/unit)
k = np.linspace(-10, 10, 4000)


def cft(f, x, k):
    """Numerical CFT: f̂(k) = ∫ f(x) e^{-ikx} dx  (trapezoidal rule)."""
    F = np.array([np.trapezoid(f * np.exp(-1j * kv * x), x) for kv in k])
    return F


# ---- (a) delta: narrow Gaussian normalised to unit area ----
sigma_d = 0.15
f_a = np.exp(-x**2 / (2 * sigma_d**2))
f_a /= np.trapezoid(f_a, x)          # unit area → f̂(k) ≈ 1
F_a = np.abs(cft(f_a, x, k))

# ---- (b) Gaussian: narrow in x → broad in k ----
sigma_b = 0.7
f_b = np.exp(-x**2 / (2 * sigma_b**2))
F_b = np.abs(cft(f_b, x, k))        # ≈ σ√(2π) e^{−k²σ²/2}

# ---- (c) Top-hat of width L ----
L = 2.0
f_c = np.where(np.abs(x) <= L / 2, 1.0, 0.0)
F_c = np.abs(cft(f_c, x, k))        # ≈ 2sin(kL/2)/k

# ---- (d) Windowed cosine at k₀ ----
k0 = 1.5
f_d = np.cos(k0 * x) * np.exp(-x**2 / 8)   # Gaussian window
F_d = np.abs(cft(f_d, x, k))

datasets = [
    (f_a, F_a,
     r"(a) $f(x) \approx \delta(x)$  (narrow Gaussian)",
     r"$|\hat{f}(k)| \approx 1$  (flat)"),
    (f_b, F_b,
     r"(b) $f(x) = e^{-x^2/(2\sigma^2)}$,  $\sigma=0.7$",
     r"$|\hat{f}(k)| = \sigma\sqrt{2\pi}\,e^{-k^2\sigma^2/2}$"),
    (f_c, F_c,
     r"(c) $f(x) = \mathrm{rect}(x/L)$,  $L=2$",
     r"$|\hat{f}(k)| = |2\sin(kL/2)/k|$"),
    (f_d, F_d,
     r"(d) $f(x) = \cos(k_0 x)$ (windowed), $k_0=1.5$",
     r"$|\hat{f}(k)|$: two spikes at $\pm k_0$"),
]

colours = ["#2166ac", "#d6604d", "#1a9850", "#762a83"]

fig = plt.figure(figsize=(9, 7.5))
# leave extra top margin for column headers
gs = GridSpec(4, 2, figure=fig, hspace=0.65, wspace=0.38,
              left=0.09, right=0.97, top=0.91, bottom=0.06)

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

    ax_r.plot(k, F, color=colours[i], lw=1.3)
    ax_r.set_xlim(-8, 8)
    ax_r.set_xlabel(r"$k$", fontsize=8)
    ax_r.set_ylabel(r"$|\hat{f}(k)|$", fontsize=8)
    ax_r.set_title(flab, fontsize=8, loc='left')
    ax_r.axhline(0, color='k', lw=0.5)
    ax_r.tick_params(labelsize=7)

# Column headers placed well above the top of the grid
fig.text(0.27, 0.975, "Real space  $f(x)$",
         ha='center', va='top', fontsize=10, fontweight='bold')
fig.text(0.73, 0.975, "Wavenumber space  $|\\hat{f}(k)|$",
         ha='center', va='top', fontsize=10, fontweight='bold')

fig.savefig("cft_pairs.pdf", bbox_inches="tight")
fig.savefig("cft_pairs.png", dpi=150, bbox_inches="tight")
print("Saved: cft_pairs.pdf, cft_pairs.png")
