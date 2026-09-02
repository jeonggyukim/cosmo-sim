#!/usr/bin/env python3
"""
plot_ngenic_seedtable.py — how the N-GenIC white noise generator lays out its
random numbers, on a small 2D example.

Transcribes the seed-table construction and the per-column draw loop of
monofonIC/src/plugins/random_ngenic.cc (the SeedTable_ fill, and Fill_Grid),
to show why one seed gives the same large-scale modes at any grid resolution.

The seed table is filled in square shells growing outward from the origin of
the transverse index plane, so the position of a given (n_x, n_y) in the master
random stream depends only on its shell, never on N. Filling the table in the
obvious raster order would destroy that, which the second panel row shows.
"""
import numpy as np
import matplotlib.pyplot as plt


def seedtable_order(N):
    """Shell index and draw order per (i, j), exactly as random_ngenic.cc fills it.

    Shell i writes eight symmetric entries: index i and its reflection N-1-i in
    each axis, and the transpose of each. The ring at radius i holds 8i+4 cells,
    matching 4*i + 4*(i+1) from the two loop lengths.
    """
    shell = np.full((N, N), -1)
    order = np.full((N, N), -1)
    counter = 0

    def put(a, b, i):
        nonlocal counter
        if shell[a, b] < 0:
            shell[a, b], order[a, b] = i, counter
        counter += 1

    for i in range(N // 2):
        for j in range(i):     put(i, j, i)
        for j in range(i + 1): put(j, i, i)
        for j in range(i):     put(N - 1 - i, j, i)
        for j in range(i + 1): put(N - 1 - j, i, i)
        for j in range(i):     put(i, N - 1 - j, i)
        for j in range(i + 1): put(j, N - 1 - i, i)
        for j in range(i):     put(N - 1 - i, N - 1 - j, i)
        for j in range(i + 1): put(N - 1 - j, N - 1 - i, i)
    return shell, order


def signed(N):
    """Array index -> signed wavenumber, the representative with smallest |k|."""
    n = np.arange(N)
    return np.where(n < N // 2, n, n - N)


def show(ax, M, N, title, cmap, cbar_label, fig, annotate=False, vmax=None):
    """Plot a per-(i,j) quantity in signed wavenumber coordinates."""
    s = signed(N)
    o = np.argsort(s)
    im = ax.imshow(M[np.ix_(o, o)], origin="lower", cmap=cmap, vmin=0,
                   vmax=vmax if vmax is not None else M.max(),
                   extent=[s[o][0] - .5, s[o][-1] + .5, s[o][0] - .5, s[o][-1] + .5])
    step = max(1, N // 8)
    ax.set_xticks(s[o][::step])
    ax.set_yticks(s[o][::step])
    ax.set_xlabel(r"$n_x$", fontsize=10)
    ax.set_ylabel(r"$n_y$", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    if annotate:
        for a in range(N):
            for b in range(N):
                ax.text(s[b], s[a], str(M[a, b].astype(int)), ha="center",
                        va="center", fontsize=6.5, color="w")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return im


N1, N2 = 8, 16
shell1, order1 = seedtable_order(N1)
shell2, order2 = seedtable_order(N2)

fig = plt.figure(figsize=(11.5, 8.8))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.42)

# --- top row: the seed table, and its independence of N --------------------
ax = fig.add_subplot(gs[0, 0])
show(ax, order1.astype(float), N1, r"(a)  order the seeds are drawn, $N=8$",
     "magma", "draw number", fig, annotate=True)

ax = fig.add_subplot(gs[0, 1])
show(ax, shell1.astype(float), N1, r"(b)  shell index, $N=8$", "viridis",
     "shell $i$", fig, vmax=N2 // 2 - 1)

ax = fig.add_subplot(gs[0, 2])
show(ax, shell2.astype(float), N2, r"(c)  shell index, $N=16$", "viridis",
     "shell $i$", fig, vmax=N2 // 2 - 1)
ax.plot([-4.5, 3.5, 3.5, -4.5, -4.5], [-4.5, -4.5, 3.5, 3.5, -4.5],
        "r-", lw=1.6)
ax.text(3.3, 4.2, r"shared with $N=8$", color="r", fontsize=8, ha="right")

# --- bottom left: shell ordering vs raster ordering ------------------------
s1, s2 = signed(N1), signed(N2)
pick = lambda M, s, N: np.array([[M[np.where(s == a)[0][0], np.where(s == b)[0][0]]
                                  for b in s1] for a in s1])
d_shell = pick(shell2, s2, N2) - pick(shell1, s1, N1)
raster = lambda N: np.arange(N * N).reshape(N, N)
d_raster = pick(raster(N2), s2, N2) - pick(raster(N1), s1, N1)

ax = fig.add_subplot(gs[1, 0])
v = np.abs(d_raster).max()
im = ax.imshow(d_raster, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
               extent=[-4.5, 3.5, -4.5, 3.5])
ax.set_xlabel(r"$n_x$", fontsize=10)
ax.set_ylabel(r"$n_y$", fontsize=10)
ax.tick_params(labelsize=8)
ax.set_title("(d)  raster order: draw position\n"
             r"shifts with $N$ (max $|\Delta| = $" f"{v:.0f})", fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label(r"$\Delta$ draw number", fontsize=9)
cb.ax.tick_params(labelsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.imshow(d_shell, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
          extent=[-4.5, 3.5, -4.5, 3.5])
ax.set_xlabel(r"$n_x$", fontsize=10)
ax.set_ylabel(r"$n_y$", fontsize=10)
ax.tick_params(labelsize=8)
ax.set_title("(e)  shell order: the same mode is\n"
             "drawn at the same position", fontsize=10)
ax.text(-0.5, -0.5, "identical\n" r"max $|\Delta| = 0$", ha="center",
        va="center", fontsize=11, color="0.2")

# --- bottom right: the per-column stream -----------------------------------
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(r"(f)  one RNG stream per $(n_x, n_y)$, run down $k_z$", fontsize=10)
for row, (lab, colour, nz) in enumerate([(r"$N=8$", "C0", 5), (r"$N=16$", "C3", 9)]):
    y = 0.74 - 0.30 * row
    ax.text(0.0, y + 0.14, lab + r":  reseed from SeedTable$[n_x,n_y]$",
            fontsize=8.5, color=colour)
    w = 0.94 / nz
    for m in range(nz):
        ax.add_patch(plt.Rectangle((0.02 + w * m, y - 0.02), w * 0.88, 0.13,
                                   fc="none", ec=colour, lw=1.2))
        ax.text(0.02 + w * (m + 0.44), y + 0.045, f"{m}", ha="center",
                va="center", fontsize=7, color=colour)
    ax.text(0.02, y - 0.075, r"$k_z = 0 \ldots N/2$, one (phase, amplitude) each",
            fontsize=7, color="0.4")
ax.annotate("", xy=(0.05, 0.72), xytext=(0.05, 0.47),
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
ax.text(0.0, 0.30,
        "Same seed and same draw order, so the modes\n"
        r"$k_z = 0\ldots4$ carry identical phase and amplitude"
        "\n"
        r"in both. The finer grid reads further down the"
        "\n"
        "same stream, adding small-scale power beneath\n"
        "modes that are already fixed.",
        fontsize=8, va="top", linespacing=1.6)

fig.suptitle("The N-GenIC white noise layout, on an $8^2$ and a $16^2$ grid",
             fontsize=12, y=0.975)
fig.savefig("ngenic_seedtable.pdf", bbox_inches="tight")
fig.savefig("ngenic_seedtable.png", dpi=150, bbox_inches="tight")
print("Saved: ngenic_seedtable.pdf, ngenic_seedtable.png")
print(f"shell order identical on shared modes: {np.abs(d_shell).max() == 0}")
print(f"raster order differs by up to {np.abs(d_raster).max()} draws")
