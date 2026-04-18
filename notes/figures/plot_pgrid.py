#!/usr/bin/env python3
"""
plot_pgrid.py — Compare P(k) and P^grid(k) for several box sizes.

P^grid(k) is computed via the convolution picture (Pen 1997, eq. 1):
  1. Compute xi(r) from P(k) via FFTLog (mcfit).
  2. Truncate xi(r) at r = L/2 (spherical approx. to cubical box).
  3. Numerically integrate back to get P^grid(k) at chosen k values,
     including k < k_f where P(k) -> 0 but P^grid(k) remains finite.

Hankel pair (l=0):
  xi(r) = 1/(2pi^2) int P(k) sinc(kr) k^2 dk
  P(k)  = 4pi       int xi(r) sinc(kr) r^2 dr   (sinc(x) = sin(x)/x)

CLASS output: ../class_pk_z45_pk.dat  (k in h/Mpc, P in (Mpc/h)^3)
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mcfit import P2xi

trapz = np.trapezoid   # numpy >= 2.0

# ---------------------------------------------------------------------------
# Load CLASS P(k)
# ---------------------------------------------------------------------------
data = np.loadtxt('../../data/class_pk_z45_pk.dat', comments='#')
k_class, P_class = data[:, 0], data[:, 1]   # h/Mpc, (Mpc/h)^3

Pk_interp = interp1d(np.log(k_class), np.log(P_class),
                     kind='cubic', bounds_error=False,
                     fill_value=-np.inf)


def P_of_k(k):
    return np.exp(Pk_interp(np.log(np.clip(k, k_class[0], k_class[-1]))))


# ---------------------------------------------------------------------------
# Compute xi(r) via FFTLog
# ---------------------------------------------------------------------------
print("Computing xi(r) via FFTLog...")
r_xi, xi_vals = P2xi(k_class, l=0, deriv=0)(P_class, extrap=True)
xi_of_r = interp1d(np.log(r_xi), xi_vals, kind='cubic',
                   bounds_error=False, fill_value=0.0)


def xi(r):
    return xi_of_r(np.log(np.clip(r, r_xi[0], r_xi[-1])))

# ---------------------------------------------------------------------------
# P^grid(k, L) via direct integration over [0, L/2]
# P^grid(k) = 4pi int_0^{L/2} xi(r) sinc(kr) r^2 dr
# ---------------------------------------------------------------------------


def compute_Pgrid(k_arr, L, Nr=3000):
    r_int = np.linspace(1e-3, L / 2.0, Nr)
    xi_int = xi(r_int)
    Pg = np.zeros_like(k_arr, dtype=float)
    for i, k in enumerate(k_arr):
        x = k * r_int
        sinc = np.where(np.abs(x) < 1e-8, 1.0, np.sin(x) / x)
        Pg[i] = 4 * np.pi * trapz(xi_int * sinc * r_int**2, r_int)
    return Pg


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
box_sizes = [25, 100, 500]
colors = ['C1', 'C2', 'C3']
linestyles = ['-', '--', '-.']

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
ax1, ax2 = axes

# P(k) from CLASS
ax1.loglog(k_class, P_class, 'k-', lw=1.8,
           label=r'$P(k)$ (CLASS, $z=45$)', zorder=5)

for L, col, ls in zip(box_sizes, colors, linestyles):
    kf = 2 * np.pi / L

    # k grid: from kf/10 up to knyq of class, spanning sub-fundamental modes
    k_sub = np.logspace(np.log10(kf / 10), np.log10(kf * 0.99), 30)
    k_sup = np.logspace(np.log10(kf),      np.log10(1.0),        60)
    k_eval = np.concatenate([k_sub, k_sup])

    print(f"Computing P^grid for L={L} Mpc/h ...")
    Pg = compute_Pgrid(k_eval, L)

    # Left panel: full curves
    ax1.loglog(k_eval[Pg > 0], Pg[Pg > 0], color=col, ls=ls, lw=1.5,
               label=rf'$P^{{\rm grid}}(k),\ L={L}\ h^{{-1}}$Mpc')
    ax1.axvline(kf, color=col, lw=0.7, alpha=0.4)

    # Right panel: ratio — meaningful for k where P(k) is not tiny
    # Show from kf/10 to show the divergence as k -> 0
    ratio = Pg / P_of_k(k_eval) - 1.0
    ax2.semilogx(k_eval, ratio, color=col, ls=ls, lw=1.5,
                 label=rf'$L={L}\ h^{{-1}}$Mpc')
    ax2.axvline(kf, color=col, lw=0.7, alpha=0.4,
                label=rf'$k_f\ (L={L})$' if L == 25 else None)

ax1.set_xlabel(r'$k$  [$h$ Mpc$^{-1}$]', fontsize=12)
ax1.set_ylabel(r'$P(k)$  [(Mpc/$h$)$^3$]', fontsize=12)
ax1.set_title(r'$P(k)$ and $P^{\rm grid}(k)$', fontsize=11)
ax1.legend(fontsize=8.5, loc='lower left')
ax1.set_xlim(1e-3, k_class[-1])
ax1.set_ylim(1e-5, None)

ax2.axhline(0.0, color='k', lw=1.0, ls=':', zorder=0)
ax2.set_xlabel(r'$k$  [$h$ Mpc$^{-1}$]', fontsize=12)
ax2.set_ylabel(r'$P^{\rm grid}(k)\,/\,P(k) - 1$', fontsize=12)
ax2.set_title(r'Fractional excess $P^{\rm grid}/P - 1$', fontsize=11)
ax2.legend(fontsize=8, loc='upper right', ncol=2)
ax2.set_xlim(1e-3, 1.0)
ax2.set_ylim(-1, 1)

fig.tight_layout()
fig.savefig('pgrid_comparison.pdf', bbox_inches='tight')
fig.savefig('pgrid_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: pgrid_comparison.pdf / .png")
