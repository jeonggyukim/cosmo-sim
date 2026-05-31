#!/usr/bin/env python3
"""
plot_meyer.py — Meyer scaling function as used in MUSIC2 for the
coarse/fine white-noise splice.  Matches the implementation in
MUSIC2/src/math/special.hh::Meyer_scaling_function.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def meyer_1d(u):
    """1D Meyer scaling function m(u), u = |k| / k_max.

    Mirrors MUSIC2's Meyer_scaling_function(k, kmax) with the
    substitution k_norm = u * 2*pi:
        u < 1/3      : 1
        1/3 < u < 2/3 : cos[pi/2 * nu(3u - 1)],   nu = clip to [0,1]
        u > 2/3       : 0
    """
    u = np.abs(u)
    out = np.zeros_like(u)
    flat = u < 1.0 / 3.0
    roll = (u >= 1.0 / 3.0) & (u < 2.0 / 3.0)
    out[flat] = 1.0
    nu = np.clip(3.0 * u[roll] - 1.0, 0.0, 1.0)
    out[roll] = np.cos(0.5 * np.pi * nu)
    return out


def shannon_1d(u):
    return np.where(np.abs(u) < 1.0, 1.0, 0.0)


fig = plt.figure(figsize=(9, 3.6))
gs = gridspec.GridSpec(1, 2, wspace=0.32)

# ---------------------------------------------------------------
# Left: the 1D scaling function m(u)
# ---------------------------------------------------------------
ax1 = fig.add_subplot(gs[0])
u = np.linspace(0, 1.05, 1200)
ax1.plot(u, meyer_1d(u), 'C0', lw=2.0, label='Meyer')
ax1.plot(u, shannon_1d(u), 'C3', lw=1.4, ls='--', label='Shannon')

ax1.axvline(1.0/3.0, color='gray', ls=':', lw=0.8)
ax1.axvline(2.0/3.0, color='gray', ls=':', lw=0.8)
ax1.text(1.0/3.0 + 0.005, 0.05, r'$u = \frac{1}{3}$', color='gray', fontsize=9)
ax1.text(2.0/3.0 + 0.005, 0.05, r'$u = \frac{2}{3}$', color='gray', fontsize=9)

ax1.set_xlabel(r'$u = |k| / k_{\max}$', fontsize=11)
ax1.set_ylabel(r'$m_{\mathrm{Meyer}}(u)$', fontsize=11)
ax1.set_xlim(0, 1.05)
ax1.set_ylim(-0.05, 1.1)
ax1.legend(fontsize=10, loc='center right', frameon=False)
ax1.set_title('1D scaling function (per Cartesian axis)', fontsize=10)

# ---------------------------------------------------------------
# Right: 3D product weights w_coarse, w_fine along axis vs diagonal.
# w_coarse(k) = prod_a m(k_a / k_Ny^{l-1})
# w_fine(k)   = sqrt(1 - w_coarse^2)   (MUSIC2 blend, eq:meyer-weights)
# ---------------------------------------------------------------
ax2 = fig.add_subplot(gs[1])

u = np.linspace(0, 1.05, 2000)

# Axis: only one component nonzero -> w_coarse = m(u) * m(0)^2 = m(u)
w_axis_coarse = meyer_1d(u)
w_axis_fine = np.sqrt(np.clip(1.0 - w_axis_coarse**2, 0.0, 1.0))

# Diagonal: k = (u, u, u)/sqrt(3) in units of k_Ny -> each component
# is u/sqrt(3); w_coarse = m(u/sqrt(3))^3
arg = u / np.sqrt(3.0)
w_diag_coarse = meyer_1d(arg)**3
w_diag_fine = np.sqrt(np.clip(1.0 - w_diag_coarse**2, 0.0, 1.0))

ax2.plot(u, w_axis_coarse, 'C0', lw=2.0,
         label=r'$w_{\mathrm{coarse}}$, axis')
ax2.plot(u, w_axis_fine, 'C0', lw=2.0, ls='--',
         label=r'$w_{\mathrm{fine}}$, axis')
ax2.plot(u, w_diag_coarse, 'C2', lw=2.0,
         label=r'$w_{\mathrm{coarse}}$, diagonal')
ax2.plot(u, w_diag_fine, 'C2', lw=2.0, ls='--',
         label=r'$w_{\mathrm{fine}}$, diagonal')

ax2.set_xlabel(r'$|\mathbf{k}| / k_{\mathrm{Ny}}^{\,\ell-1}$', fontsize=11)
ax2.set_ylabel('weight', fontsize=11)
ax2.set_xlim(0, 1.05)
ax2.set_ylim(-0.05, 1.1)
ax2.legend(fontsize=9, loc='center left', bbox_to_anchor=(0.02, 0.42),
           frameon=False)
ax2.set_title('3D blend weights along axis vs diagonal', fontsize=10)

fig.tight_layout()
fig.savefig('meyer.pdf', bbox_inches='tight')
fig.savefig('meyer.png', dpi=150, bbox_inches='tight')
print("Saved: meyer.pdf, meyer.png")
