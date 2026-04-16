#!/usr/bin/env python3
"""
plot_pancake.py — Zel'dovich pancake collapse.

Left panel:  density profile 1+δ = 1/|1 − D_+ cos(q)| vs Eulerian position x,
             for several values of D_+.  The density diverges at the caustic
             (D_+ = 1), illustrating the pancake singularity.

Right panel: particle trajectories x(q) = q − D_+ sin(q).  For D_+ < 1 the
             map is monotone; at D_+ = 1 trajectories touch at q=0; for D_+ > 1
             the map folds, creating three streams (multi-streaming).

Perturbation: plane wave  δ_lin(q) = cos(q),  Ψ^(1)(q) = −sin(q).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'axes.linewidth' : 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
ax1, ax2 = axes

# ── parameters ─────────────────────────────────────────────────────────────
q = np.linspace(-np.pi, np.pi, 20_000)
Dvals   = [0.3, 0.6, 0.9, 1.0, 1.2]
lstyles = ['-',  '-',  '-', '--', ':']
labels  = [r'$D_+=0.3$', r'$D_+=0.6$', r'$D_+=0.9$',
           r'$D_+=1$ (caustic)', r'$D_+=1.2$ (multi-stream)']
colors  = [cm.plasma(v) for v in [0.15, 0.35, 0.60, 0.80, 0.95]]

# ── LEFT panel: density vs Eulerian x ──────────────────────────────────────
rho_max = 12.0

for D, ls, lbl, c in zip(Dvals, lstyles, labels, colors):
    x   = q - D * np.sin(q)              # Eulerian position
    J   = 1.0 - D * np.cos(q)            # Jacobian ∂x/∂q
    rho = 1.0 / np.abs(J)                # 1 + δ = 1/|J|

    # For D > 1 the map is multi-valued; split into monotone segments
    if D <= 1.0:
        # single-stream everywhere; clip the caustic spike for display
        ax1.plot(x / np.pi, np.clip(rho, 0, rho_max), color=c, ls=ls,
                 lw=1.8, label=lbl, zorder=3 if D == 1.0 else 2)
    else:
        # Outside the caustic J > 0 (single stream)
        mask = J > 0.02
        xp, rp = x.copy(), rho.copy()
        xp[~mask] = np.nan
        rp[~mask] = np.nan
        ax1.plot(xp / np.pi, np.clip(rp, 0, rho_max), color=c, ls=ls,
                 lw=1.8, label=lbl, zorder=1)
        # Inside the caustic show dashed continuation (unphysical in single-stream)
        mask2 = J < -0.02
        xp2, rp2 = x.copy(), rho.copy()
        xp2[~mask2] = np.nan
        rp2[~mask2] = np.nan
        ax1.plot(xp2 / np.pi, np.clip(rp2, 0, rho_max), color=c, ls=':',
                 lw=1.0, alpha=0.5, zorder=1)

ax1.axhline(1.0, color='gray', lw=0.7, ls='--', alpha=0.6)
ax1.set_xlim(-1, 1)
ax1.set_ylim(0, rho_max)
ax1.set_xlabel(r'Eulerian position $x\,/\,\pi$', fontsize=12)
ax1.set_ylabel(r'$1 + \delta$', fontsize=12)
ax1.set_title(r"Density profile", fontsize=12)
ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax1.text(-0.95, 11.0,
         r'$1+\delta = \dfrac{1}{|1 - D_+\cos q|}$',
         fontsize=10, va='top')
ax1.annotate('pancake\ncaustic', xy=(0, rho_max * 0.97),
             xytext=(0.25, rho_max * 0.85),
             fontsize=9, color='gray',
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# ── RIGHT panel: particle trajectories x(q) ────────────────────────────────
q_line = np.linspace(-np.pi, np.pi, 2000)

# Unperturbed lattice (D=0)
ax2.plot(q_line / np.pi, q_line / np.pi, color='k', lw=0.8,
         ls='--', label='$D_+=0$ (lattice)', alpha=0.5)

for D, ls, lbl, c in zip(Dvals, lstyles, labels, colors):
    x = q_line - D * np.sin(q_line)
    ax2.plot(q_line / np.pi, x / np.pi, color=c, ls=ls, lw=1.8, label=lbl)

# Shade multi-streaming region for D=1.2
D_ms = 1.2
# Shell-crossing occurs where J=0: q_c = ±arccos(1/D_ms)
q_cross = np.arccos(1.0 / D_ms)
x_cross_lo = q_cross - D_ms * np.sin(q_cross)
x_cross_hi = -x_cross_lo
ax2.axhspan(x_cross_hi / np.pi, x_cross_lo / np.pi,
            alpha=0.08, color='purple',
            label=r'multi-stream zone ($D_+=1.2$)')

ax2.set_xlim(-1, 1)
ax2.set_ylim(-1.4, 1.4)
ax2.set_xlabel(r'Lagrangian position $q\,/\,\pi$', fontsize=12)
ax2.set_ylabel(r'Eulerian position $x\,/\,\pi$', fontsize=12)
ax2.set_title(r'Particle trajectories  $x = q - D_+\sin q$', fontsize=12)
ax2.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
ax2.axhline(0, color='gray', lw=0.5)
ax2.axvline(0, color='gray', lw=0.5)
ax2.text(0.52, -1.25,
         'fold-over\n(shell crossing)',
         fontsize=8.5, ha='center', color='purple')

# ── layout and save ─────────────────────────────────────────────────────────
fig.suptitle(r"Zel'dovich pancake: plane-wave perturbation $\delta_\mathrm{lin}(q)=\cos q$",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig('pancake.pdf', bbox_inches='tight')
fig.savefig('pancake.png', dpi=150, bbox_inches='tight')
print("Saved: pancake.pdf, pancake.png")
