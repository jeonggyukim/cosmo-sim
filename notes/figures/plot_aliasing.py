#!/usr/bin/env python3
"""
plot_aliasing.py — Figure illustrating aliasing for fft.tex.

N=8 equally spaced samples on [0,1], Delta=1/8.
True signal:   5 cycles in [0,1]  →  k = 5·(2π/Δ), above k_Nyq = 4·(2π/Δ)
Aliased signal: 3 cycles in [0,1] →  k' = k − 2π/Δ = −3·(2π/Δ)
Both pass through identical sample values.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 8
Delta = 1.0 / N
x_cont = np.linspace(0, 1, 600)
x_samp = np.arange(N) * Delta

y_true  = np.cos(2 * np.pi * 5 * x_cont)   # 5 cycles (above Nyquist)
y_alias = np.cos(2 * np.pi * 3 * x_cont)   # 3 cycles (alias)
y_dots  = np.cos(2 * np.pi * 5 * x_samp)   # sample values (same on both)

fig, ax = plt.subplots(figsize=(7, 2.8))

ax.plot(x_cont, y_true,  color='#2166ac', lw=1.5, ls='--',
        label=r'true signal: $k=5\cdot(2\pi/\Delta)$, above $k_{\rm Nyq}=4\cdot(2\pi/\Delta)$')
ax.plot(x_cont, y_alias, color='#d6604d', lw=1.5, ls='-',
        label=r'aliased signal: $k\'=k-2\pi/\Delta=-3\cdot(2\pi/\Delta)$')
ax.plot(x_samp, y_dots,  'o', color='black', ms=5, zorder=5,
        label=r'sample values $f_n$ (identical on both curves)')

ax.axhline(0, color='k', lw=0.5)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r'$x$', fontsize=11)
ax.set_ylabel(r'$f(x)$', fontsize=11)

# x-tick labels as x_0 … x_7
ax.set_xticks(x_samp)
ax.set_xticklabels([fr'$x_{n}$' for n in range(N)], fontsize=8)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
          ncol=1, fontsize=8, frameon=True, handlelength=2.5)

fig.tight_layout()
fig.savefig("aliasing.pdf", bbox_inches="tight")
fig.savefig("aliasing.png", dpi=150, bbox_inches="tight")
print("Saved: aliasing.pdf, aliasing.png")
