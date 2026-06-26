"""
Figure for fft.tex sec:per_mode_dist: why |n_hat(k)|^2 is exponentially
distributed.

Three panels:
  (a) Scatter of (Re[n_hat], Im[n_hat]) at a fixed non-special mode for
      M independent realisations of an N x N white-noise grid.  The
      cloud is an isotropic 2D Gaussian with variance sigma^2 = N^2/2
      per component (we normalise out N^2 so the cloud has variance 1/2).
  (b) Histogram of R = sqrt(Re^2 + Im^2) with the Rayleigh PDF
      overlaid.
  (c) Histogram of R^2 = Re^2 + Im^2 with the exponential PDF overlaid.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

N = 64
M = 20000
mode = (3, 5)

samples = np.empty(M, dtype=complex)
for i in range(M):
    n = rng.standard_normal((N, N))
    nhat = np.fft.fft2(n)
    samples[i] = nhat[mode]

Ntot = N * N
samples /= np.sqrt(Ntot)
re = samples.real
im = samples.imag
R = np.abs(samples)
Rsq = R**2

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

ax = axes[0]
ax.scatter(re, im, s=2, alpha=0.25, color='C0', rasterized=True)
theta = np.linspace(0, 2*np.pi, 200)
for r in [1.0, 2.0]:
    ax.plot(r*np.cos(theta), r*np.sin(theta), 'k--', lw=0.7,
            label=f'R={r:.0f}' if r == 1.0 else None)
ax.set_aspect('equal')
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_xlabel(r'$\mathrm{Re}[\hat n] / \sqrt{N_{\rm tot}}$')
ax.set_ylabel(r'$\mathrm{Im}[\hat n] / \sqrt{N_{\rm tot}}$')
ax.set_title('(a) Isotropic 2D Gaussian')
ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)

ax = axes[1]
bins = np.linspace(0, 3.5, 50)
ax.hist(R, bins=bins, density=True, alpha=0.6, color='C0', label='samples')
r_grid = np.linspace(0, 3.5, 200)
sigma2 = 0.5
rayleigh = (r_grid / sigma2) * np.exp(-r_grid**2 / (2*sigma2))
ax.plot(r_grid, rayleigh, 'r-', lw=1.5, label=r'Rayleigh: $r/\sigma^2 e^{-r^2/2\sigma^2}$')
ax.set_xlabel(r'$R = |\hat n| / \sqrt{N_{\rm tot}}$')
ax.set_ylabel('density')
ax.set_title('(b) Rayleigh: $R$')
ax.legend(loc='upper right', fontsize=8)

ax = axes[2]
bins = np.linspace(0, 8, 50)
ax.hist(Rsq, bins=bins, density=True, alpha=0.6, color='C0', label='samples')
u_grid = np.linspace(0, 8, 200)
mean_Rsq = 1.0
exp_pdf = np.exp(-u_grid / mean_Rsq) / mean_Rsq
ax.plot(u_grid, exp_pdf, 'r-', lw=1.5,
        label=r'Exp: $\frac{1}{\mu} e^{-u/\mu}$, $\mu=1$')
ax.set_xlabel(r'$R^2 = |\hat n|^2 / N_{\rm tot}$')
ax.set_ylabel('density')
ax.set_yscale('log')
ax.set_ylim(1e-3, 2)
ax.set_title('(c) Exponential: $R^2$')
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('grf_exponential.pdf', dpi=150)
print('wrote grf_exponential.pdf')
