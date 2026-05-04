"""Figure: 2D restriction of delta — real-space, Fourier maps, mode-ratio vs sinc^2."""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
})

rng = np.random.default_rng(42)
N, N2, L = 64, 128, 1.0
n_pk = -2.0

kx = np.fft.fftfreq(N2, d=L/N2) * 2*np.pi
KX, KY = np.meshgrid(kx, kx, indexing='ij')
Kmag = np.sqrt(KX**2 + KY**2)
Pk   = np.where(Kmag > 0, Kmag**n_pk, 0.0)

white   = rng.standard_normal((N2, N2))
delta_k = np.fft.fftn(white) * np.sqrt(Pk); delta_k[0,0] = 0.0
delta_2N = np.fft.ifftn(delta_k).real
delta_2N -= delta_2N.mean()
delta_N  = delta_2N.reshape(N, 2, N, 2).mean(axis=(1, 3))

dk_2N = np.fft.fftn(delta_2N) / N2**2
dk_N  = np.fft.fftn(delta_N)  / N**2
kc = np.fft.fftfreq(N, d=L/N) * 2*np.pi
KXc, KYc = np.meshgrid(kc, kc, indexing='ij')
Kc = np.sqrt(KXc**2 + KYc**2)
nc = np.fft.fftfreq(N, d=1/N).astype(int)
ix = np.where(nc >= 0, nc, nc + N2)
dk_2N_at_coarse = dk_2N[np.ix_(ix, ix)]
ratio = np.abs(dk_N) / (np.abs(dk_2N_at_coarse) + 1e-30)

dx_f = L / N2

fig, axarr = plt.subplots(2, 3, figsize=(15, 9))
ax = axarr.flatten()

vmax = max(abs(delta_2N).max(), abs(delta_N).max())
real_ext = [0, L, 0, L]
im = ax[0].imshow(delta_2N, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  origin='lower', extent=real_ext)
ax[0].set_title(r'$\delta_{2N}$ (fine)')
ax[0].set_xlabel('x'); ax[0].set_ylabel('y')
plt.colorbar(im, ax=ax[0], fraction=0.045)
im = ax[1].imshow(delta_N, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  origin='lower', extent=real_ext)
ax[1].set_title(r'$\delta_N$ (restricted)')
ax[1].set_xlabel('x'); ax[1].set_ylabel('y')
plt.colorbar(im, ax=ax[1], fraction=0.045)

def shifted_log(d):
    return np.log10(np.abs(np.fft.fftshift(d)) + 1e-12)
kfund = 2*np.pi / L
kNy_f = np.pi / dx_f  # fine Nyquist; used to normalize all Fourier axes
fourier_ext = [-1, 1, -1, 1]
im = ax[2].imshow(shifted_log(dk_2N), cmap='viridis', origin='lower',
                  extent=fourier_ext)
ax[2].set_title(r'$\log_{10}|\hat\delta_{2N}(k)|$')
ax[2].set_xlabel(r'$k_x / k_{\mathrm{Ny},2N}$')
ax[2].set_ylabel(r'$k_y / k_{\mathrm{Ny},2N}$')
plt.colorbar(im, ax=ax[2], fraction=0.045)

Kflat = Kc.ravel() / kNy_f; Rflat = ratio.ravel()
W2_2d = (np.cos(KXc*dx_f/2) * np.cos(KYc*dx_f/2))**2  # exact 2D W^2
W2flat = W2_2d.ravel()
m = Kflat > 0
ax[3].semilogy(Kflat[m], Rflat[m], '.', ms=3, alpha=0.4,
               color='C0', label='measured')
# Bin W^2 in |k| and shade min/max as a band: at fixed |k|, axis-aligned
# vs diagonal modes feel different cell-window suppression.
nbins_W = 40
kmax_W = Kflat[m].max()
Wbins = np.linspace(0, kmax_W, nbins_W+1)
Widx  = np.digitize(Kflat[m], Wbins) - 1
kmid, Wmin, Wmax = [], [], []
for i in range(nbins_W):
    sel = Widx == i
    if sel.sum() >= 2:
        kmid.append(0.5*(Wbins[i]+Wbins[i+1]))
        Wmin.append(W2flat[m][sel].min())
        Wmax.append(W2flat[m][sel].max())
kmid = np.array(kmid); Wmin = np.array(Wmin); Wmax = np.array(Wmax)
ax[3].fill_between(kmid, Wmin, Wmax, color='green', alpha=0.3,
                   label=r'$W^2(k_x,k_y)$ band (no aliases)')
ks_tilde = np.linspace(0, Kflat.max(), 400)
ax[3].semilogy(ks_tilde, np.cos(ks_tilde*np.pi/2)**2, 'r-',
               label=r'$\cos^2(\pi \tilde k / 2)$ (1D proxy)')
ax[3].set_xlabel(r'$|k_c| / k_{\mathrm{Ny},2N}$')
ax[3].set_ylabel(r'$|\hat\delta_N|/|\hat\delta_{2N}|$')
ax[3].set_ylim(0.1, 10)

# Panel 5: binned P_N / P_2N ratio with theory prediction.
# Need P_2N at all fine modes and P_N at coarse modes (per-mode |dk|^2).
P_2N = (np.abs(dk_2N)**2)
P_N  = (np.abs(dk_N )**2)
# Analytic alias-sum theory (no realization dependence):
# <|dk_N(k_c)|^2> = sum_n W^2(k_c + pi n / dx_f) * P(k_c + pi n / dx_f),
# where P(k') = |k'|^n_pk is the input power-law spectrum.
# We then divide by P(k_c) = |k_c|^n_pk to form the ratio plotted in
# panel 4.
theory_PN_analytic = np.zeros_like(P_N)
for nx in (0, 1):
    for ny in (0, 1):
        ix_shift = (ix + N*nx) % N2
        iy_shift = (ix + N*ny) % N2
        KX_at = kx[ix_shift][:, None]
        KY_at = kx[iy_shift][None, :]
        Kmag_at = np.sqrt(KX_at**2 + KY_at**2)
        W2_at = (np.cos(KX_at*dx_f/2) * np.cos(KY_at*dx_f/2))**2
        with np.errstate(divide='ignore'):
            P_at = np.where(Kmag_at > 0, Kmag_at**n_pk, 0.0)
        theory_PN_analytic += W2_at * P_at
P_input_at_coarse = np.where(Kc > 0, Kc**n_pk, np.inf)

# Bin both PN and theory_PN by |k_c|.
def radial_mean(field, K, nbins=20, kmax=None):
    K = K.ravel(); F = field.ravel()
    m = K > 0
    K = K[m]; F = F[m]
    if kmax is None: kmax = K.max()
    bins = np.linspace(0, kmax, nbins+1)
    idx = np.digitize(K, bins) - 1
    Fbar = np.zeros(nbins); kbar = np.zeros(nbins); cnt = np.zeros(nbins)
    for i in range(nbins):
        sel = idx == i
        if sel.any():
            Fbar[i] = F[sel].mean()
            kbar[i] = K[sel].mean()
            cnt[i]  = sel.sum()
    keep = cnt > 0
    return kbar[keep], Fbar[keep]

kbar, PN_bar     = radial_mean(P_N,      Kc)
_,    P2N_at_bar = radial_mean(P_2N[np.ix_(ix, ix)], Kc)

# Smooth analytic alias-sum prediction: evaluate on a dense (k_x, k_y)
# Brillouin-zone grid (no need for the coarse-mode lattice), then bin
# radially with many bins. Independent of the random realization.
def analytic_aliassum_ratio(n_pk, kny_coarse, kny_fine, ngrid=512, nbins=200):
    # Square coarse Brillouin zone, k_x, k_y in [-kny_coarse, kny_coarse].
    # Radial |k_c| extends up to the corner kny_coarse * sqrt(2).
    kc_grid = np.linspace(-kny_coarse, kny_coarse, ngrid)
    KXg, KYg = np.meshgrid(kc_grid, kc_grid, indexing='ij')
    Kg = np.sqrt(KXg**2 + KYg**2)
    num = np.zeros_like(Kg)
    for nx in (0, 1):
        for ny in (0, 1):
            kx_a = KXg + nx*2*kny_coarse
            ky_a = KYg + ny*2*kny_coarse
            Kmag_a = np.sqrt(kx_a**2 + ky_a**2)
            W2_a = (np.cos(kx_a*np.pi/(2*kny_fine)) *
                    np.cos(ky_a*np.pi/(2*kny_fine)))**2
            with np.errstate(divide='ignore'):
                P_a = np.where(Kmag_a > 0, Kmag_a**n_pk, 0.0)
            num += W2_a * P_a
    with np.errstate(divide='ignore'):
        denom = np.where(Kg > 0, Kg**n_pk, np.inf)
    ratio = num / denom
    # Stop at the axis-aligned coarse Nyquist: above k_Ny,N the radial
    # ring is entirely inside the BZ corners (diagonal-only modes), and
    # the bin average drops abruptly because W^2 is smallest there.
    # Truncating avoids that geometric artefact and matches how the
    # measured curve is typically interpreted.
    kmax = kny_coarse
    bins = np.linspace(0, kmax, nbins+1)
    idx = np.digitize(Kg.ravel(), bins) - 1
    rflat = ratio.ravel()
    kbins = 0.5*(bins[:-1] + bins[1:])
    rbar = np.full(nbins, np.nan)
    for i in range(nbins):
        sel = idx == i
        if sel.any():
            rbar[i] = rflat[sel].mean()
    keep = ~np.isnan(rbar)
    return kbins[keep], rbar[keep]

kny_coarse = np.pi * N / L
k_smooth, ratio_smooth = analytic_aliassum_ratio(n_pk, kny_coarse, kNy_f)

# Overlay smooth analytic prediction on bottom-left panel.
ax[3].semilogy(k_smooth/kNy_f, ratio_smooth, '-', color='red', lw=1.5,
               label=r'analytic $\langle P_N\rangle/P(k_c)$ (alias sum)')
ax[3].legend(fontsize=8, loc='lower left')

ax[4].plot(kbar/kNy_f, PN_bar/P2N_at_bar, 'o-', color='C0',
           label=r'measured (single realization)')
ax[4].plot(k_smooth/kNy_f, ratio_smooth, '-', color='red', lw=1.5,
           label=r'analytic: $\sum_{\bf n}|W|^2 P(k_c{+}\pi{\bf n}/\Delta x_f)\,/\,P(k_c)$')
ax[4].axhline(1, color='gray', lw=0.5)
ax[4].set_xlabel(r'$|k_c| / k_{\mathrm{Ny},2N}$')
ax[4].set_ylabel(r'$P_N(k_c) / P_{2N}(k_c)$')
ax[4].set_title('binned power ratio (alias-sum check)')
ax[4].legend()

# Bottom-right: smooth analytic alias-sum prediction P_N/P_2N for
# several spectral slopes n.
n_compare = [0.0, -2.0, -5.0]
colors    = ['C0', 'C1', 'C3']
for n_val, col in zip(n_compare, colors):
    k_n, ratio_n = analytic_aliassum_ratio(n_val, kny_coarse, kNy_f)
    ax[5].plot(k_n/kNy_f, ratio_n, '-', color=col, lw=1.5,
               label=fr'$n={n_val:g}$')
# W^2 band (axis-aligned vs body-diagonal extremes) for reference.
ax[5].fill_between(kmid, Wmin, Wmax, color='green', alpha=0.3,
                   label=r'$W^2(k_x,k_y)$ band')
ax[5].axhline(1, color='gray', lw=0.5)
ax[5].set_xlabel(r'$|k_c| / k_{\mathrm{Ny},2N}$')
ax[5].set_ylabel(r'$P_N(k_c) / P_{2N}(k_c)$')
ax[5].set_title(r'alias-sum ratio vs slope $n$')
ax[5].set_yscale('log')
ax[5].set_ylim(0.05, 10)
ax[5].legend(fontsize=9, loc='lower left')

plt.tight_layout()
plt.savefig('restriction_density.pdf')
print("wrote restriction_density.pdf")
