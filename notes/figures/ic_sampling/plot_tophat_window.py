#!/usr/bin/env python3
"""
plot_tophat_window.py — Plot the spherical top-hat window function
in real space and Fourier space.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 2, wspace=0.35)

# ---------------------------------------------------------------
# Left panel: real-space top-hat W_T(r/R)
# ---------------------------------------------------------------
ax1 = fig.add_subplot(gs[0])
x = np.linspace(0, 2.5, 500)
W_real = np.where(x <= 1.0, 1.0, 0.0)
ax1.plot(x, W_real, 'C0', lw=2)
ax1.axvline(1.0, color='gray', ls='--', lw=0.8)
ax1.set_xlabel(r'$r/R$', fontsize=12)
ax1.set_ylabel(r'$W_T(r/R)$', fontsize=12)
ax1.set_title('Real space', fontsize=11)
ax1.set_xlim(0, 2.5)
ax1.set_ylim(-0.1, 1.3)
ax1.text(1.05, 0.55, r'$r = R$', color='gray', fontsize=10)
ax1.set_yticks([0, 0.5, 1.0])

# ---------------------------------------------------------------
# Right panel: Fourier-space top-hat Ŵ_T(kR)
# ---------------------------------------------------------------
ax2 = fig.add_subplot(gs[1])

u = np.linspace(1e-4, 15, 2000)   # u = kR
# Ŵ_T(u) = 3[sin(u) - u*cos(u)] / u^3
W_fourier = 3.0 * (np.sin(u) - u * np.cos(u)) / u**3

ax2.plot(u, W_fourier, 'C1', lw=2, label=r'$\hat{W}_T(kR)$')
ax2.axhline(0, color='gray', lw=0.6, ls='--')

# Mark first zero crossing at u = tan(u), numerically ~4.493
ax2.axvline(np.pi, color='C2', ls=':', lw=1.0, label=r'$kR=\pi$')

ax2.set_xlabel(r'$kR$', fontsize=12)
ax2.set_ylabel(r'$\hat{W}_T(kR)$', fontsize=12)
ax2.set_title('Fourier space', fontsize=11)
ax2.set_xlim(0, 15)
ax2.set_ylim(-0.25, 1.05)
ax2.legend(fontsize=10, loc='upper right')

# Annotation for the formula
ax2.text(5.5, 0.55,
         r'$\dfrac{3[\sin(kR) - kR\cos(kR)]}{(kR)^3}$',
         fontsize=10, ha='left', va='center')

fig.suptitle('Spherical top-hat window function $W_T$', fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig('tophat_window.pdf', bbox_inches='tight')
fig.savefig('tophat_window.png', dpi=150, bbox_inches='tight')
print("Saved: tophat_window.pdf, tophat_window.png")
