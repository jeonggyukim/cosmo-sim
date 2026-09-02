#!/usr/bin/env python3
"""Merge sweep directories produced by parallel runs of pencil_seed_sweep.py.

Running several sweeps at once is the efficient way to use the machine, since
only the monofonIC step is threaded and the measurement is single-threaded. Each
run needs its own --out directory and a disjoint seed range; this script then
presents them to the analysis scripts as one sweep.

    python pencil_seed_sweep.py --seed0 5001 --nseeds 500 --nthreads 1 --out DIR_A &
    python pencil_seed_sweep.py --seed0 5501 --nseeds 500 --nthreads 1 --out DIR_B &
    wait
    python merge_sweeps.py --out DIR_ALL DIR_A DIR_B

The merged directory holds a copy of theory.hdf5, a rebuilt summary.hdf5, and a
symbolic link to every seed directory, so nothing is duplicated on disk and
analyze_sweep.py, plot_sweep_summary.py and plot_deviation_stats.py all work on
it unchanged.

The runs must agree on everything that defines the measurement -- grid, box,
pencil geometry, species, k bins. A mismatch is an error rather than a warning,
because averaging spectra measured on different bins is silently wrong.
"""
import argparse, glob, os, shutil
import numpy as np, h5py

CHECK_ATTRS = ["N", "L", "frac", "npen", "lperp", "dkperp", "kny", "kf", "fvol", "npencils"]


def theory_signature(path):
    with h5py.File(path) as f:
        attrs = {a: f.attrs[a] for a in CHECK_ATTRS if a in f.attrs}
        species = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
        return attrs, species, f["k"][:], f["P_theory"][:], f["P_win"][:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="sweep directories to merge")
    ap.add_argument("--out", required=True, help="merged directory to create")
    ap.add_argument("--copy", action="store_true",
                    help="copy the seed directories instead of linking to them")
    a = ap.parse_args()

    dirs = [d.rstrip(os.sep) for d in a.dirs]
    for d in dirs:
        if not os.path.exists(f"{d}/theory.hdf5"):
            raise SystemExit(f"{d} has no theory.hdf5 -- is it a sweep directory?")

    ref = theory_signature(f"{dirs[0]}/theory.hdf5")
    for d in dirs[1:]:
        cur = theory_signature(f"{d}/theory.hdf5")
        if cur[0] != ref[0] or cur[1] != ref[1]:
            raise SystemExit(f"{d} was run with different parameters from {dirs[0]}:\n"
                             f"   {dirs[0]}: {ref[0]}, species {ref[1]}\n"
                             f"   {d}: {cur[0]}, species {cur[1]}")
        for name, x, y in [("k", ref[2], cur[2]), ("P_theory", ref[3], cur[3]),
                           ("P_win", ref[4], cur[4])]:
            if not np.allclose(x, y):
                raise SystemExit(f"{d} disagrees with {dirs[0]} on {name}")

    os.makedirs(a.out, exist_ok=True)
    shutil.copy(f"{dirs[0]}/theory.hdf5", f"{a.out}/theory.hdf5")

    seen, linked = {}, 0
    for d in dirs:
        for src in sorted(glob.glob(f"{d}/seed_*")):
            name = os.path.basename(src)
            if name in seen:
                raise SystemExit(f"{name} appears in both {seen[name]} and {d}. "
                                 f"Parallel sweeps need disjoint seed ranges.")
            seen[name] = d
            dst = os.path.join(a.out, name)
            if os.path.lexists(dst):
                os.remove(dst) if os.path.islink(dst) else shutil.rmtree(dst)
            if a.copy:
                shutil.copytree(src, dst)
            else:
                os.symlink(os.path.abspath(src), dst)
            linked += 1

    rows, cols = [], None
    for d in dirs:
        s = f"{d}/summary.hdf5"
        if not os.path.exists(s):
            continue
        with h5py.File(s) as f:
            cols = cols or [k for k in f]
            rows.append(np.stack([f[k][:] for k in cols], 1))
            note, fit = f.attrs.get("note", b""), (f.attrs.get("fit_kmin"), f.attrs.get("fit_kmax"))
    if rows:
        allrows = np.concatenate(rows)
        with h5py.File(f"{a.out}/summary.hdf5", "w") as f:
            for n, key in enumerate(cols):
                f[key] = allrows[:, n]
            f.attrs["species"] = np.array(ref[1], dtype=h5py.string_dtype())
            if fit[0] is not None:
                f.attrs["fit_kmin"], f.attrs["fit_kmax"] = fit
            f.attrs["note"] = note
            f.attrs["merged_from"] = np.array([os.path.abspath(d) for d in dirs],
                                              dtype=h5py.string_dtype())

    print(f"merged {len(dirs)} sweeps -> {a.out}")
    print(f"  {linked} realizations, {ref[0].get('npencils', '?')} pencils each, "
          f"species {ref[1]}")
    print(f"  seed directories are {'copies' if a.copy else 'symlinks'}; "
          f"theory.hdf5 copied from {dirs[0]}")


if __name__ == "__main__":
    main()
