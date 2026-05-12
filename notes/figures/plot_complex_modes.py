"""
Figure for fft.tex: phasor plots of e^{+ikx} and e^{-ikx} sampled on
an N=8 grid, comparing a sub-Nyquist case (distinct phasors, opposite
rotation direction) with the Nyquist case (collapse onto the same
alternating +/-1 pattern).
"""
import numpy as np
import matplotlib.pyplot as plt

N = 8
n = np.arange(N)
xn = n / N            # x_n = n*Delta with Delta = 1/N (L=1)
theta = np.linspace(0, 2*np.pi, 200)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))


def phasor_panel(ax, k, title):
    phi_pos = np.exp(+1j * k * xn * 2 * np.pi)
    phi_neg = np.exp(-1j * k * xn * 2 * np.pi)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.7, alpha=0.5)
    for shift in [-1, 0, 1]:
        ax.axhline(shift, color='k', lw=0.2, alpha=0.3)
        ax.axvline(shift, color='k', lw=0.2, alpha=0.3)
    for i in range(N - 1):
        dx = phi_pos[i + 1].real - phi_pos[i].real
        dy = phi_pos[i + 1].imag - phi_pos[i].imag
        ax.annotate('', xy=(phi_pos[i + 1].real, phi_pos[i + 1].imag),
                    xytext=(phi_pos[i].real, phi_pos[i].imag),
                    arrowprops=dict(arrowstyle='->', color='C0', lw=1.2))
    for i in range(N - 1):
        dx = phi_neg[i + 1].real - phi_neg[i].real
        dy = phi_neg[i + 1].imag - phi_neg[i].imag
        ax.annotate('', xy=(phi_neg[i + 1].real, phi_neg[i + 1].imag),
                    xytext=(phi_neg[i].real, phi_neg[i].imag),
                    arrowprops=dict(arrowstyle='->', color='C3', lw=1.2,
                                    linestyle='--'))
    ax.scatter(phi_pos.real, phi_pos.imag, c='C0', s=70, zorder=5,
               label=r'$e^{+ikx_n}$ (CCW)')
    ax.scatter(phi_neg.real, phi_neg.imag, c='C3', s=30, zorder=6,
               marker='x', label=r'$e^{-ikx_n}$ (CW)')
    for i in range(N):
        ax.annotate(f'{i}',
                    xy=(phi_pos[i].real, phi_pos[i].imag),
                    xytext=(1.18 * phi_pos[i].real, 1.18 * phi_pos[i].imag),
                    fontsize=8, ha='center', va='center', color='C0')
    ax.set_aspect('equal')
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel(r'$\mathrm{Re}$')
    ax.set_ylabel(r'$\mathrm{Im}$')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9)


# Sub-Nyquist: k = N/4 = k_Nyq/2 (so e^{i k x_n} = e^{i 2*pi*n/4})
phasor_panel(axes[0], N / 4,
             r'(a) Sub-Nyquist: $k = k_{\rm Nyq}/2$.  Phasors rotate'
             '\nCCW for $+k$, CW for $-k$ -- distinct sequences.')

# Nyquist: k = N/2 = k_Nyq (so e^{i k x_n} = e^{i pi n} = (-1)^n)
phasor_panel(axes[1], N / 2,
             r'(b) Nyquist: $k = k_{\rm Nyq}$.  Both $\pm k$ collapse'
             '\nto $(-1)^n$ on the grid -- phasors overlap.')

plt.tight_layout()
plt.savefig('complex_modes.pdf', dpi=150)
print('wrote complex_modes.pdf')
