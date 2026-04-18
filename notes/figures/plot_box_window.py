#!/usr/bin/env python3
"""
plot_box_window.py — Illustrate the cubical box window function W_C
in real space and Fourier space (1D slices for clarity).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 2, wspace=0.38)

L = 1.0   # normalise box side to 1

# ---------------------------------------------------------------
# Left: real-space W_C(x) — 1D slice along one axis
# ---------------------------------------------------------------
ax1 = fig.add_subplot(gs[0])
x = np.linspace(-1.0, 1.0, 1000)
W_real = np.where(np.abs(x) <= L/2, 1.0, 0.0)
ax1.plot(x, W_real, 'C0', lw=2)
ax1.axvline(-L/2, color='gray', ls='--', lw=0.8)
ax1.axvline(L/2, color='gray', ls='--', lw=0.8)
ax1.set_xlabel(r'$x_i / L$', fontsize=12)
ax1.set_ylabel(r'$W_C(x_i)$', fontsize=12)
ax1.set_title('Real space (1D slice)', fontsize=11)
ax1.set_xlim(-1.0, 1.0)
ax1.set_ylim(-0.15, 1.3)
ax1.text(-0.5, 1.08, r'$-L/2$', ha='center', color='gray', fontsize=9)
ax1.text(0.5, 1.08, r'$+L/2$', ha='center', color='gray', fontsize=9)
ax1.set_yticks([0, 0.5, 1.0])

# ---------------------------------------------------------------
# Right: Fourier-space W_C(k) — 1D sinc along one axis
# ---------------------------------------------------------------
ax2 = fig.add_subplot(gs[1])

# k in units of 2pi/L = Delta_k
u = np.linspace(-6.5, 6.5, 2000)   # u = k_i * L/2  (dimensionless)
# sinc(k_i L/2) = sin(k_i L/2) / (k_i L/2)
sinc_vals = np.where(np.abs(u) < 1e-10, 1.0, np.sin(u) / u)

# The full W_C(k) = L^3 * prod sinc(k_i L/2); plot 1D factor L*sinc
ax2.plot(u / np.pi, sinc_vals, 'C1', lw=2,
         label=r'$\mathrm{sinc}(k_i L/2)$')
ax2.axhline(0, color='gray', lw=0.6, ls='--')

# Mark Delta_k = 2pi/L, i.e. u = pi
for n in [-2, -1, 1, 2]:
    ax2.axvline(n, color='C2', ls=':', lw=0.8, alpha=0.7)
ax2.axvline(0, color='C2', ls=':', lw=0.8, alpha=0.7,
            label=r'$k_i = n\,\Delta k$')

ax2.set_xlabel(r'$k_i\,/\,\Delta k$  $(\Delta k = 2\pi/L)$', fontsize=12)
ax2.set_ylabel(r'$\mathrm{sinc}(k_i L/2)$', fontsize=12)
ax2.set_title('Fourier space (1D factor)', fontsize=11)
ax2.set_xlim(-6.5/np.pi, 6.5/np.pi)
ax2.set_ylim(-0.35, 1.15)
ax2.legend(fontsize=9, loc='upper right')

# Annotate width ~ Delta_k
ax2.annotate('', xy=(1, -0.28), xytext=(-1, -0.28),
             arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))
ax2.text(0, -0.33, r'width $\sim \Delta k$', ha='center', fontsize=9)

fig.suptitle(r'Box window function $W_C$', fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig('box_window.pdf', bbox_inches='tight')
fig.savefig('box_window.png', dpi=150, bbox_inches='tight')
print("Saved: box_window.pdf, box_window.png")
