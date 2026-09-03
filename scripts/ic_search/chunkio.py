#!/usr/bin/env python3
"""Read a sweep in either layout and return the per-pencil quantities.

pencil_seed_sweep.py writes one directory per seed for small local runs and one
chunk file per batch for cluster arrays. Every analysis needs the same arrays out
of both, so the reading lives here rather than in each script.
"""
import glob
import numpy as np, h5py


def shape_params(lam):
    """Ellipticity and prolateness of the tidal ellipsoid, from sorted eigenvalues.

    The textbook definitions divide by the trace, which passes through zero for a
    region of average density and makes the ratio diverge. Dividing by the norm
    sqrt(sum l_i^2) keeps the same meaning and stays finite everywhere.
    """
    l1, l2, l3 = lam[..., 0], lam[..., 1], lam[..., 2]
    L = np.sqrt((lam**2).sum(-1)) + 1e-30
    return (l1 - l3)/(2*L), (l1 - 2*l2 + l3)/(2*L)


def load(path, species="matter", nchunk=None, want_spectra=False):
    """Return (crit_theory, crit_window, cols, meta, spectra).

    cols maps a quantity name to an array of shape (nseed, npencil). The two
    criteria are the distance of each pencil from the raw theory and from the
    theory convolved with the pencil window, over the low-k band.
    """
    chunks = sorted(glob.glob(f"{path}/chunk_*.hdf5"))
    seeds = sorted(glob.glob(f"{path}/seed_*/pk.hdf5"))
    if not chunks and not seeds:
        raise SystemExit(f"no chunk_*.hdf5 or seed_*/pk.hdf5 under {path}")
    if nchunk:
        chunks = chunks[:nchunk]

    ref = chunks[0] if chunks else f"{path}/theory.hdf5"
    with h5py.File(ref) as f:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
        SP = names.index(species)
        k, P_th, P_win = f["k"][:], f["P_theory"][SP], f["P_win"][SP]
        meta = {a: f.attrs[a] for a in ("N", "L", "kny", "dkperp") if a in f.attrs}
    lo = k <= 2*meta["dkperp"]
    hi = (k > 2*meta["dkperp"]) & (k <= 0.9*meta["kny"])
    meta.update(k=k, P_th=P_th, P_win=P_win, lo=lo, hi=hi)

    ct, cw, cols, spec, spec_box = [], [], {}, [], []

    def add(name, arr):
        cols.setdefault(name, []).append(arr)

    def take_chunk(f, P):
        """P is (nseed, npencil, nk). Read each dataset once and slice in memory:
        indexing HDF5 per seed turns this into millions of tiny reads."""
        ct.append(np.sqrt((np.log(P[:, :, lo]/P_th[lo])**2).mean(2)))
        cw.append(np.sqrt((np.log(P[:, :, lo]/P_win[lo])**2).mean(2)))
        add("large-scale power", (P[:, :, lo]/P_win[lo]).mean(2))
        add("small-scale power", (P[:, :, hi]/P_win[hi]).mean(2))
        if want_spectra:
            spec.append(P)
            # The whole box of the same realizations, so a figure can show what
            # the estimator does without a mask beside what it does with one.
            if "P_full" in f:
                pf = f["P_full"]
                spec_box.append(pf[:, SP] if pf.ndim == 3 else pf[SP][None])
        if "shear" not in f:
            return
        RS = f["smooth_R"][:]
        shear, dbar = f["shear"][:], f["dbar"][:]
        contrast = f["contrast"][:] if "contrast" in f else None
        lam = f["lambda"][:] if "lambda" in f else None
        web = f["webtype"][:] if "webtype" in f else None
        for r, R in enumerate(RS):
            add(f"tidal shear R={R:.0f}", shear[:, r])
            add(f"mean overdensity R={R:.0f}", dbar[:, r])
            if contrast is not None:
                add(f"env contrast R={R:.0f}", contrast[:, r])
            if lam is not None:
                e, _ = shape_params(lam[:, r])
                add(f"ellipticity R={R:.0f}", e)
            if web is not None:
                for w, wn in enumerate(("knot", "filament", "sheet", "void")):
                    add(f"{wn} fraction R={R:.0f}", web[:, r, :, w])
        if "bulk" in f:
            add("bulk flow", np.linalg.norm(f["bulk"][:], axis=-1))

        # sigma(R) on a top-hat, the amplitude the definition of sigma_8 uses.
        # Named by radius so a figure can say sigma_8 rather than "the rms at
        # R = 8". The whole-box value is one number per realization, broadcast
        # across the pencil axis to sit beside the per-pencil columns.
        if "sigma" in f and "sigma_R" in f:
            SR = f["sigma_R"][:]
            sig = f["sigma"][:]
            for r, R in enumerate(SR):
                add(f"sigma R={R:.0f}", sig[:, r])
            if "sigma_box" in f:
                sb = f["sigma_box"][:]
                for r, R in enumerate(SR):
                    add(f"sigma box R={R:.0f}",
                        np.repeat(sb[:, r, None], P.shape[1], 1))

        # The same quantities on the pencil trimmed by a margin, and on the whole
        # box. Both are one value per seed for the box, so they are broadcast to
        # the pencil axis to sit alongside the per-pencil columns.
        npen = P.shape[1]
        for src, label in (("shear_interior", "tidal shear interior"),
                           ("dbar_interior", "mean overdensity interior")):
            if src in f:
                arr = f[src][:]
                for r, R in enumerate(RS):
                    add(f"{label} R={R:.0f}", arr[:, r])
        for src, label in (("shear_box", "tidal shear box"),
                           ("dbar_box", "mean overdensity box")):
            if src in f:
                arr = f[src][:]
                for r, R in enumerate(RS):
                    add(f"{label} R={R:.0f}", np.repeat(arr[:, r, None], npen, 1))
        if "webtype_box" in f:
            wb = f["webtype_box"][:]
            for r, R in enumerate(RS):
                for w, wn in enumerate(("knot", "filament", "sheet", "void")):
                    add(f"{wn} fraction box R={R:.0f}",
                        np.repeat(wb[:, r, w, None], npen, 1))

    if chunks:
        for fn in chunks:
            with h5py.File(fn) as f:
                take_chunk(f, f["P_pencil"][:, SP])
    else:
        for fn in seeds:
            with h5py.File(fn) as f:
                take_chunk(f, f["P_pencil"][SP][None])

    ct = np.concatenate(ct); cw = np.concatenate(cw)
    cols = {n: np.concatenate(v) for n, v in cols.items()}
    if want_spectra:
        # The box spectra ride along in meta rather than in the return tuple, so
        # that adding them does not change the signature every caller unpacks.
        meta["P_box"] = np.concatenate(spec_box) if spec_box else None
    return (ct, cw, cols, meta,
            np.concatenate(spec) if want_spectra else None)


# The quantity every selection figure plots, named once here so the figures can
# show its definition rather than leaving the reader to infer it from an axis
# label. Q is whichever property is being measured; the average runs over the
# subvolumes the criterion retains, and sigma_Q over all of them, so a shift of 1
# means the selection moved Q by one standard deviation of its own scatter.
SHIFT_SYMBOL = r"$\Delta_Q$"
SHIFT_DEF = (r"$\Delta_Q \equiv \left(\langle Q\rangle_{\rm selected}"
             r" - \langle Q\rangle_{\rm all}\right)/\sigma_Q$")

# What the bars mean, which is not what the axis unit means. sigma_Q is the
# spread between regions and sets the scale of the axis; the bar is the
# uncertainty on the plotted shift, and shrinks as the sweep grows. Both live in
# the same units, so a figure that shows one without naming the other invites
# them to be read as the same thing.
ERRBAR_DEF = ("error bars: bootstrap over realizations, resampling whole seeds "
              "with their 24 subvolumes together")


def annotate_shift(fig, x=0.005, y=0.005, fontsize=10.5, ax=None):
    """Write the definition of the shift below the axes that plot it.

    Passing `ax` puts the text under that panel's left edge. Without it the text
    goes to the figure's bottom left, which is only right when the whole figure
    plots the shift: in a figure whose left panel plots something else, a
    definition parked under that panel appears to define it.
    """
    if ax is not None:
        x = ax.get_position().x0
    fig.text(x, y, SHIFT_DEF + "\n" + ERRBAR_DEF, fontsize=fontsize,
             color="0.35", ha="left", va="bottom", linespacing=1.5)


def plot_signed(ax, x, y, **kw):
    """Draw xi on a log axis as |xi|, solid where positive and dashed where not.

    A log axis cannot show a negative number, and xi turns over near 80 Mpc/h,
    so plotting it directly drops everything past the zero crossing. The
    convention here is the one icpipe/cli/plot_ic.py already uses for the same
    quantity: the magnitude is drawn throughout and the line style carries the
    sign. Callers must therefore not spend line style on anything else in these
    panels; colour and marker are free.

    Any label is attached to the positive branch alone, so a legend gains one
    entry per curve rather than two.
    """
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(y) & (x > 0)
    pos, neg = ok & (y > 0), ok & (y < 0)
    kw.pop("ls", None), kw.pop("linestyle", None)
    if pos.any():
        ax.plot(x[pos], y[pos], ls="-", **kw)
    if neg.any():
        kw.pop("label", None)
        ax.plot(x[neg], -y[neg], ls="--", **kw)


def selection_stats(crit, cols, keep):
    """Helpers for asking what a selection does, shared by the plotting scripts.

    Returns (shift, shift_err, radii_for). shift(name, crit_key) is the mean of
    the kept subvolumes minus the mean of all, in units of the scatter over all.
    shift_err bootstraps over realizations, which is the independent unit:
    subvolumes inside one box share that box's modes.

    crit is a dict of criterion arrays keyed by name, each of shape
    (nseed, npencil); cols holds the quantities in the same shape.
    """
    import re
    nseed, npen = next(iter(crit.values())).shape
    C = {k: v.ravel() for k, v in crit.items()}
    Q = {n: v.ravel() for n, v in cols.items()}
    rng = np.random.default_rng(0)

    def shift(name, which, idx=None):
        T, c = Q[name], C[which]
        if idx is not None:
            T, c = T[idx], c[idx]
        n = max(1, int(round(keep*len(c))))
        sel = np.argpartition(c, n)[:n]
        sd = T.std()
        return (T[sel].mean() - T.mean())/sd if sd > 0 else np.nan

    def shift_err(name, which, nboot=200):
        boot = np.empty(nboot)
        for b in range(nboot):
            g = rng.integers(0, nseed, nseed)
            boot[b] = shift(name, which, (g[:, None]*npen + np.arange(npen)).ravel())
        return float(np.std(boot))

    def radii_for(stem):
        """Smoothing radii present for one quantity, sorted, as (radius, key)."""
        out = []
        for n in Q:
            m = re.fullmatch(rf"{stem} R=(\d+(?:\.\d+)?)", n)
            if m:
                out.append((float(m.group(1)), n))
        return sorted(out)

    return shift, shift_err, radii_for


# A sphere of radius R at the mean matter density encloses this mass, which says
# which objects a smoothing scale corresponds to. Omega_m = 0.3.
MASS_PER_R3 = 4*np.pi/3 * 0.3 * 2.775e11    # h^-1 Msun per (h^-1 Mpc)^3
