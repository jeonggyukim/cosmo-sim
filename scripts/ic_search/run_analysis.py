#!/usr/bin/env python3
"""Run every analysis and figure for one sweep, into a single timestamped folder.

The measurements a sweep produces are read by half a dozen scripts, each with
its own options, and running them by hand invites a figure made from one cut and
a table made from another. This runs them together against one directory, with
one retained fraction, and puts the figures and the text output side by side so
a folder is a self-contained record of what the data said.

    python run_analysis.py --data DIR [--keep 0.01] [--out DIR]

A step that fails does not stop the others; the summary at the end says which
ran. Steps whose inputs are missing from a given sweep, such as the whole-box
quantities in an early run, fail here rather than being silently skipped.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "big128"),
                help="sweep directory holding chunk_*.hdf5 or seed_*/")
ap.add_argument("--keep", type=float, default=0.01,
                help="fraction of subvolumes the selection retains")
ap.add_argument("--out", default=None,
                help="output directory (default: a timestamped one under the "
                     "figures root, named after the sweep)")
ap.add_argument("--python", default=sys.executable)
A = ap.parse_args()

if not os.path.isdir(A.data):
    raise SystemExit(f"no such sweep directory: {A.data}")

tag = os.path.basename(A.data.rstrip("/"))
OUT = A.out or os.path.join(paths.FIGS, f"{tag}_{time.strftime('%Y%m%d_%H%M')}")
os.makedirs(OUT, exist_ok=True)

K = str(A.keep)
STEPS = [
    ("shift table, matched against the raw theory",
     ["analyze_environment.py", "--data", A.data, "--keep", K],
     "shifts_raw_theory.txt"),
    ("shift table, matched against the convolved theory",
     ["analyze_environment.py", "--data", A.data, "--keep", K, "--reference", "window"],
     "shifts_windowed_control.txt"),
    ("shift table, one subvolume per realization",
     ["analyze_environment.py", "--data", A.data, "--keep", K, "--one-per-seed", "40"],
     "shifts_one_per_seed.txt"),
    ("summary figure",
     ["plot_selection_summary.py", "--data", A.data, "--keep", K,
      "--out", os.path.join(OUT, "selection_summary.png")],
     "selection_summary.txt"),
    ("distributions figure",
     ["plot_selection_histograms.py", "--data", A.data, "--keep", K,
      "--out", os.path.join(OUT, "selection_histograms.png")],
     "selection_histograms.txt"),
    ("shift against search size",
     ["plot_search_size.py", "--data", A.data,
      "--out", os.path.join(OUT, "search_size.png")],
     "search_size.txt"),
    ("correlations between the measured quantities",
     ["plot_correlations.py", "--data", A.data,
      "--out", os.path.join(OUT, "correlations.png")],
     "correlations.txt"),
    ("amplitude fitted with and without the window",
     ["fit_amplitude.py", "--data", A.data, "--keep", K,
      "--out", os.path.join(OUT, "amplitude_fit.png")],
     "amplitude_fit.txt"),
    ("dependence on smoothing scale, and the box comparison",
     ["plot_scale_and_box.py", "--data", A.data, "--keep", K,
      "--out", os.path.join(OUT, "scale_and_box.png")],
     "scale_and_box.txt"),
]

print(f"sweep  {A.data}")
print(f"output {OUT}")
print(f"keep   {100*A.keep:g}%\n")

results = []
for name, argv, logname in STEPS:
    t0 = time.time()
    proc = subprocess.run([A.python, os.path.join(HERE, argv[0])] + argv[1:],
                          cwd=HERE, capture_output=True, text=True)
    with open(os.path.join(OUT, logname), "w") as f:
        f.write(proc.stdout)
        if proc.stderr.strip():
            f.write("\n--- stderr ---\n" + proc.stderr)
    ok = proc.returncode == 0
    results.append((name, ok, logname))
    print(f"  {'ok  ' if ok else 'FAIL'} {name}  ({time.time()-t0:.1f} s)")
    if not ok:
        tail = [l for l in proc.stderr.strip().splitlines() if l.strip()][-2:]
        for line in tail:
            print(f"       {line}")

nfail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results)-nfail}/{len(results)} steps ran")
print(f"figures and logs in {OUT}")
sys.exit(1 if nfail else 0)
