#!/usr/bin/env python3
"""
plot_growth.py — Linear growth factor D_+(z) and growth rates f(z), f^(2)(z).

Left panel:  D_+(z) normalized to D_+(z=0)=1, for flat ΛCDM (Ωm=0.3, ΩΛ=0.7).
             Dashed reference: matter-domination limit D_+ ∝ 1/(1+z).

Right panel: f(z) ≈ Ωm(a)^0.55 (Linder 2005) and f^(2)(z) ≈ 2 Ωm(a)^(4/7)
             (Bouchet+1995).  Dashed references: matter-dom limits f→1, f^(2)→2.

Growth factor computed via the integral formula:
    D_+(a) ∝ H(a) ∫_0^a  da' / [a' H(a')]^3
using scipy.integrate.quad.
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ── cosmology ──────────────────────────────────────────────────────────────
Om0 = 0.3
OmL = 0.7
H0 = 67.32        # km/s/Mpc (fiducial; only ratios matter here)


def H(a):
    """Hubble parameter H(a) / H0."""
    return np.sqrt(Om0 * a**-3 + OmL)


def Om_a(a):
    """Omega_m(a)."""
    return Om0 * a**-3 / H(a)**2

# Growth factor via integral  D_+(a) ∝ H(a) ∫_0^a da'/[a'H(a')]^3


def D_integrand(ap):
    return 1.0 / (ap * H(ap))**3


def D_plus(a):
    integral, _ = quad(D_integrand, 1e-6, a, limit=200)
    return H(a) * integral


# Normalise to D_+(a=1) = 1
D_norm = D_plus(1.0)

# ── redshift grid ───────────────────────────────────────────────────────────
z = np.logspace(np.log10(0.01), np.log10(50), 400)
a = 1.0 / (1.0 + z)

D_arr = np.array([D_plus(ai) / D_norm for ai in a])
Om_arr = Om_a(a)
f_arr = Om_arr ** 0.55                        # Linder (2005)
f2_arr = 2.0 * Om_arr ** (4.0/7.0)            # Bouchet+ (1995)

# Matter-dom references
D_matdom = 1.0 / (1.0 + z) / (1.0 / (1.0 + 0.01))   # normalized at z=0.01

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# ── LEFT: D_+(z) ────────────────────────────────────────────────────────────
ax1.loglog(z, D_arr, 'C0', lw=2.0, label=r'$D_+(z)$, flat $\Lambda$CDM')
# Matter-dom reference D_+ ∝ 1/(1+z)  — normalize to match at high z
D_md_ref = D_arr[-1] * (1 + z[-1]) / (1 + z)   # scale from high-z anchor
ax1.loglog(z, D_md_ref, 'k--', lw=1.0, alpha=0.6,
           label=r'$\propto (1+z)^{-1}$ (matter-dom)')

# Mark z=0 and a few reference redshifts
for z_ref, lbl in [(0.01, r'$z\approx 0$'), (2, r'$z=2$'), (10, r'$z=10$')]:
    idx = np.argmin(np.abs(z - z_ref))
    ax1.axvline(z_ref, color='gray', lw=0.6, ls=':', alpha=0.5)
    ax1.text(z_ref * 1.15, D_arr[idx] * 1.15, lbl, fontsize=8, color='gray')

# Shade Λ-dominated regime (rough: a > 0.5, z < 1)
ax1.axvspan(0.01, 1.0, alpha=0.06, color='C3')
ax1.text(0.15, 0.03, r'$\Lambda$-dominated', fontsize=8.5,
         color='C3', rotation=0, va='center')

ax1.set_xlim(z.min(), z.max())
ax1.set_ylim(3e-3, 1.5)
ax1.set_xlabel(r'Redshift $z$', fontsize=12)
ax1.set_ylabel(r'$D_+(z)\;/\;D_+(0)$', fontsize=12)
ax1.set_title(r'Linear growth factor', fontsize=12)
ax1.legend(fontsize=9)

# ── RIGHT: f(z) and f^(2)(z) ────────────────────────────────────────────────
ax2.semilogx(z, f_arr,  'C0', lw=2.0,
             label=r'$f(z) = \Omega_{\rm m}(z)^{0.55}$')
ax2.semilogx(z, f2_arr, 'C1', lw=2.0,
             label=r'$f^{(2)}(z) = 2\,\Omega_{\rm m}(z)^{4/7}$')

# Matter-domination limits
ax2.axhline(1.0, color='C0', lw=0.9, ls='--', alpha=0.6,
            label=r'$f\to 1$ (matter-dom)')
ax2.axhline(2.0, color='C1', lw=0.9, ls='--', alpha=0.6,
            label=r'$f^{(2)}\to 2$ (matter-dom)')

# Shading
ax2.axvspan(0.01, 1.0, alpha=0.06, color='C3')
ax2.text(0.15, 0.08, r'$\Lambda$-dominated', fontsize=8.5,
         color='C3', rotation=0, va='center')

ax2.set_xlim(z.min(), z.max())
ax2.set_ylim(0, 2.25)
ax2.set_xlabel(r'Redshift $z$', fontsize=12)
ax2.set_ylabel(r'Growth rate', fontsize=12)
ax2.set_title(r'Logarithmic growth rates', fontsize=12)
ax2.legend(fontsize=9, loc='lower right')

# Annotate relation f = d ln D_+ / d ln a
ax2.text(0.35, 1.55,
         r'$f \equiv \dfrac{d\ln D_+}{d\ln a}$',
         fontsize=10, ha='center',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))

# ── layout and save ─────────────────────────────────────────────────────────
fig.suptitle(r'Flat $\Lambda$CDM: $\Omega_{\rm m}=0.3$, $\Omega_\Lambda=0.7$',
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig('growth.pdf', bbox_inches='tight')
fig.savefig('growth.png', dpi=150, bbox_inches='tight')
print("Saved: growth.pdf, growth.png")
