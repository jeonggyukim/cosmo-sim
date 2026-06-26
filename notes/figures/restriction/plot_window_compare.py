"""Figure: discrete 2-sample averaging kernel vs continuous coarse-cell top-hat.

Shows both the real-space shapes and their Fourier-space window functions, so
the reader can see where the exact discrete W (cos) agrees with the continuous
top-hat limit (sinc) and where they diverge near the coarse Nyquist.

All horizontal axes are in units of the fine spacing dx_f, so dx_c = 2.
"""
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

# --- Real-space comparison (both centered on origin, convention A) ---
# Centered discrete kernel: stems at x = ±0.5 dx_f, weight 1/2 each. Sum = 1.
ax[0].stem([-0.5, 0.5], [0.5, 0.5], linefmt='C0-', markerfmt='C0o', basefmt=' ',
           label=r'centered discrete kernel (weight)')
# Continuous top-hat: width dx_c = 2 dx_f, height 1/dx_c = 1/2 in these units,
# centered on origin -> spans [-1, 1].
xt = np.array([-1.0, -1.0, 1.0, 1.0])
yt = np.array([ 0.0,  0.5, 0.5, 0.0])
ax[0].plot(xt, yt, 'r-', lw=2,
           label=r'continuous top-hat (density $1/\Delta x_c$)')
ax[0].axvline(0.0, color='gray', ls=':', lw=1, label='shared centroid (origin)')
ax[0].set_xlim(-1.7, 1.7)
ax[0].set_ylim(-0.05, 0.7)
ax[0].set_xlabel(r'$x / \Delta x_f$')
ax[0].set_ylabel('kernel value')
ax[0].set_title(r'real space (axes in units of $\Delta x_f$)')
ax[0].legend(fontsize=8, loc='upper right')

# --- Fourier-space comparison (extend past fine Nyquist to show side lobes) ---
u = np.linspace(0, 2*np.pi, 800)
W      = np.cos(u/2)
W_cont = np.where(u > 0, np.sin(u)/u, 1.0)
ax[1].plot(u, W,      'C0-', lw=2,
           label=r'$W = \cos(k\Delta x_f/2)$  (centered discrete)')
ax[1].plot(u, W_cont, 'r--', lw=2,
           label=r'$W_{\rm cont} = \mathrm{sinc}(k\Delta x_c/2)$')
ax[1].axvline(np.pi/2, color='gray',  ls=':', lw=1,
              label=r'coarse Nyquist ($u = \pi/2$)')
ax[1].axvline(np.pi,   color='black', ls=':', lw=1,
              label=r'fine Nyquist ($u = \pi$)')
ax[1].axhline(0, color='gray', lw=0.4)
ax[1].set_xlim(0, 2*np.pi)
ax[1].set_ylim(-0.5, 1.05)
ax[1].set_xlabel(r'$u \equiv k\Delta x_f$  (dimensionless)')
ax[1].set_ylabel('window value')
ax[1].set_title('Fourier space (1D)')
ax[1].legend(fontsize=8, loc='upper right')

# Annotate the values at the coarse Nyquist.
print(f"At k Δx_f = π/2  (coarse Nyquist):")
print(f"   W (discrete) = cos(π/4)  = {np.cos(np.pi/4):.4f}")
print(f"   W_cont       = sinc(π/2) = {np.sin(np.pi/2)/(np.pi/2):.4f}")

plt.tight_layout()
plt.savefig('window_compare.pdf')
print("wrote window_compare.pdf")
