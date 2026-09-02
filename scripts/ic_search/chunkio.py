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

    ct, cw, cols, spec = [], [], {}, []

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
    return (ct, cw, cols, meta,
            np.concatenate(spec) if want_spectra else None)
