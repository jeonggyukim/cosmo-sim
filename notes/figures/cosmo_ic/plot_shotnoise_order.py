#!/usr/bin/env python3
"""
plot_shotnoise_order.py — order of CIC-window deconvolution and shot-noise
subtraction, demonstrated on a pure Poisson particle field.

A pure Poisson field (particles drawn independently, uniform-random) has
zero clustering signal: P_signal(k) = 0 for all k, so every non-zero
measured power is shot noise.  This isolates the shot term and shows which
ordering of the two operations — window deconvolution (divide by W^2) and
shot subtraction (subtract 1/n_bar) — is correct.

Mass assignment convolves each particle with the CIC window, so
delta_meas(k) = W(k) * sum_p exp(-i k.x_p).  The RAW periodogram shot power
is therefore W^2(k)/n_bar, NOT white.  Consequences shown:

  grey   : deconvolved, no subtraction   P^m/W^2                  -> flat ~ 1/n_bar
  green  : deconv THEN subtract          P^m/W^2 - 1/n_bar        -> flat ~ 0   (correct)
  magenta: subtract THEN deconv          (P^m - 1/n_bar)/W^2      -> negative near Nyquist (wrong)

All three are computed from the SAME raw periodogram; only the order differs.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})

# ── parameters ─────────────────────────────────────────────────────────────
L   = 100.0        # box size (arbitrary units)
Np  = 64**3        # particle count
ng  = 64           # FFT mesh
rng = np.random.default_rng(1)


def cic_deposit(pos, boxsize, ngrid):
    """Single-pass CIC deposit of unit-weight particles onto ngrid^3 mesh."""
    cell = boxsize / ngrid
    g = pos / cell
    i0 = np.floor(g).astype(int)
    dx = g - i0
    out = np.zeros((ngrid, ngrid, ngrid))
    for a in range(2):
        wx = (1 - dx[:, 0]) if a == 0 else dx[:, 0]
        ix = (i0[:, 0] + a) % ngrid
        for b in range(2):
            wy = (1 - dx[:, 1]) if b == 0 else dx[:, 1]
            iy = (i0[:, 1] + b) % ngrid
            for c in range(2):
                wz = (1 - dx[:, 2]) if c == 0 else dx[:, 2]
                iz = (i0[:, 2] + c) % ngrid
                np.add.at(out, (ix, iy, iz), wx * wy * wz)
    return out


def raw_periodogram(pos, boxsize, ngrid, interlace):
    """|delta_k|^2 * V / ng^6, the raw (pre-deconvolution) periodogram, and
    the k-space helpers (|k|, W^2)."""
    N = len(pos)
    rho = cic_deposit(pos, boxsize, ngrid)
    fk = np.fft.rfftn(rho)
    if interlace:
        h = boxsize / ngrid
        rho2 = cic_deposit((pos + h / 2) % boxsize, boxsize, ngrid)
        f2 = np.fft.rfftn(rho2)
        kf = 2 * np.pi / boxsize
        kax = np.fft.fftfreq(ngrid, d=1.0 / ngrid) * kf
        kzax = np.fft.rfftfreq(ngrid, d=1.0 / ngrid) * kf
        KX, KY, KZ = np.meshgrid(kax, kax, kzax, indexing='ij')
        phase = np.exp(1j * (KX + KY + KZ) * (h / 2))
        fk = 0.5 * (fk + phase * f2)
    # overdensity delta_k = FFT(rho)/rhobar_cell with DC removed
    fk[0, 0, 0] -= N
    dk = fk / (N / ngrid**3)
    V = boxsize**3
    Pmode = np.abs(dk)**2 * V / ngrid**6           # = (V/ng^2)|dk_sumconv|^2 with dk here already /rhobar
    # k grids and CIC squared window
    kf = 2 * np.pi / boxsize
    kax = np.fft.fftfreq(ngrid, d=1.0 / ngrid) * kf
    kzax = np.fft.rfftfreq(ngrid, d=1.0 / ngrid) * kf
    KX, KY, KZ = np.meshgrid(kax, kax, kzax, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    knyq = np.pi * ngrid / boxsize
    s = lambda k: np.sinc(k / (2 * knyq))
    W2 = (s(KX) * s(KY) * s(KZ))**4
    W2[0, 0, 0] = 1.0
    Pshot = V / N
    return Pmode, Kmag, W2, knyq, Pshot


def radial_mean(vals, K, edges):
    Kf, Vf = K.ravel(), vals.ravel()
    m = Kf > 0
    Kf, Vf = Kf[m], Vf[m]
    kc, mean = [], []
    for i in range(len(edges) - 1):
        sel = (Kf >= edges[i]) & (Kf < edges[i + 1])
        if sel.any():
            kc.append(Kf[sel].mean())
            mean.append(Vf[sel].mean())
    return np.array(kc), np.array(mean)


# ── figure ─────────────────────────────────────────────────────────────────
coords = rng.uniform(0, L, size=(Np, 3))
fig, ax = plt.subplots(figsize=(6.4, 4.4))

Pmode, K, W2, knyq, Pshot = raw_periodogram(coords, L, ng, interlace=True)
edges = np.logspace(np.log10(2 * np.pi / L), np.log10(knyq * 0.95), 16)

k, P_rawdec = radial_mean(Pmode / W2, K, edges)                 # deconv, no subtraction
_, P_orig = radial_mean(Pmode / W2, K, edges)
P_orig = P_orig - Pshot                                          # deconv -> subtract (correct)
_, P_fix = radial_mean((Pmode - Pshot) / W2, K, edges)          # subtract -> deconv (wrong)

x = k / knyq
ax.axhline(0, color='k', lw=0.8, ls=':')
ax.plot(x, P_rawdec / Pshot, 'o-', color='0.6', ms=4,
        label=r'$P^m/W^2$ (no subtraction) $\to \approx 1$')
ax.plot(x, P_orig / Pshot, 's-', color='#1b7837', ms=5,
        label=r'$P^m/W^2 - 1/\bar n$  (deconv then subtract) $\to \approx 0$')
ax.plot(x, P_fix / Pshot, '^-', color='#c2276f', ms=5,
        label=r'$(P^m - 1/\bar n)/W^2$  (subtract then deconv)')

ax.set_xlabel(r'$k / k_{\rm Ny}$')
ax.set_ylabel(r'$\hat P(k)\,/\,(1/\bar n)$')
ax.set_ylim(-2.6, 1.9)
ax.set_xlim(0, 0.9)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
ax.set_title(r'Pure Poisson field ($P_{\rm signal}=0$), interlaced CIC', fontsize=10)

fig.tight_layout()
fig.savefig('shotnoise_order.pdf')
fig.savefig('shotnoise_order.png', dpi=130)
print('wrote shotnoise_order.pdf / .png')
