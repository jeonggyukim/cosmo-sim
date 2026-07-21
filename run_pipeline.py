#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end IC generation and validation pipeline.

Unified driver for two IC codes, selected with --ic-code:
  * music     — legacy multi-scale MUSIC (zoom-capable; the current default)
  * monofonic — MUSIC2-monofonIC (unigrid; adds PLT, 3LPT)

The IC front-end (build, config, run) branches on --ic-code; every downstream
step (CLASS P(k), rbins, xi, psi, P(k), plotting) is code-agnostic because both
codes write the same SWIFT HDF5 output.

Steps (each skipped if its output already exists):
  1. Build the IC code
  2. Build compute_xi / compute_xi_cic (C, Corrfunc)
  3. Generate the IC config from a template
  4. Run the IC code -> SWIFT IC HDF5 (+ CLASS ini for MUSIC)
  5. CLASS matter P(k) at z=0 (MUSIC: adapt its CLASS ini; monofonic: reuse the
     shared, cosmology-keyed data/class_pk_z0_pk.dat)
  6. Corrfunc radial bin file
  7. Corrfunc xi(r)         (skipped at z > 10; shot-noise dominated)
  8. CIC-grid xi(r), psi(r) (any z)
  9. CIC + FFT P(k)
 10. Diagnostic plot

All outputs are keyed by STEM:
  music     : n{N}_z{z}_L{L}[_nofix][_s{seed}]
  monofonic : n{N}_z{z}_L{L}_mono[_nofix][_s{seed}]
so MUSIC and monofonIC runs at the same N, z, L coexist without collision.

Note: run_pipeline.sh is the legacy bash driver (MUSIC only); it is kept as a
fallback while this Python driver is validated.
"""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from glob import glob

# Repo root is the directory holding this script.
REPO = os.path.dirname(os.path.abspath(__file__))

# CV_22 cosmology (Illustris/TNG) — matches both conf templates.
H0, OMEGA_M, OMEGA_B = "67.11", "0.3", "0.049"

# Shared, cosmology-keyed CLASS output (not stem-keyed): the theory P(k) depends
# only on cosmology, which both templates match, so a single file serves all runs.
CLASS_PKS = "data/class_pk_z0_pk.dat"


def log(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command from the repo root, raising on failure."""
    return subprocess.run(cmd, cwd=REPO, check=True, **kw)


def py_cli(module: str, *args: str) -> list[str]:
    """Invoke an icpipe CLI through the current interpreter (no PATH/conda dependency)."""
    return [sys.executable, "-m", module, *map(str, args)]


def peak_child_mib() -> float:
    """Peak RSS of child processes so far, in MiB (ru_maxrss is bytes on macOS, kB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    div = 1048576 if sys.platform == "darwin" else 1024
    return rss / div


def move_if(src_rel: str, dst_rel: str) -> bool:
    """Move REPO/src_rel to REPO/dst_rel if the source exists; report and return success."""
    src = os.path.join(REPO, src_rel)
    if not os.path.exists(src):
        return False
    shutil.move(src, os.path.join(REPO, dst_rel))
    print(f"    Saved: {dst_rel}")
    return True


# ---------------------------------------------------------------------------
# Per-code configuration
# ---------------------------------------------------------------------------

def stem_for(args) -> str:
    amp = "" if args.fix_amplitude == "yes" else "_nofix"
    sd = f"_s{args.seed}" if args.seed else ""
    mono = "_mono" if args.ic_code == "monofonic" else ""
    return f"n{args.ngrid}_z{args.zstart}_L{args.lbox}{mono}{amp}{sd}"


def ic_binary(ic_code: str) -> str:
    return "music_build/MUSIC" if ic_code == "music" else "monofonic_build/monofonIC"


def build_script(ic_code: str) -> str:
    return "tools/build-music.sh" if ic_code == "music" else "tools/build-monofonic.sh"


def conf_path(ic_code: str, stem: str) -> str:
    name = "CV_22_MUSIC" if ic_code == "music" else "CV_22_monofonIC"
    return f"conf/{name}_{stem}.conf"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_build_ic(args, stem):
    binary = ic_binary(args.ic_code)
    if os.path.exists(os.path.join(REPO, binary)):
        log(f"{args.ic_code} already built — skipping")
        return
    log(f"Building {args.ic_code}...")
    env = dict(os.environ)
    if args.ic_code == "music" and args.ic_source_dir:
        env["MUSIC2_SOURCE_DIR"] = args.ic_source_dir
    if args.ic_code == "monofonic" and args.ic_source_dir:
        env["MONOFONIC_SOURCE_DIR"] = args.ic_source_dir
    run([build_script(args.ic_code)], env=env)


def step_build_compute_xi(args):
    if os.path.exists(os.path.join(REPO, "bin/compute_xi")):
        log("compute_xi already built — skipping")
        return
    log("Building compute_xi / compute_xi_cic...")
    cmd = ["make", "-C", "src"]
    if args.corrfunc_dir:
        cmd.append(f"CORRFUNCDIR={args.corrfunc_dir}")
    run(cmd)


def step_gen_config(args, stem):
    conf = conf_path(args.ic_code, stem)
    if os.path.exists(os.path.join(REPO, conf)):
        log(f"config already exists — skipping ({conf})")
        return conf
    log(f"Generating {args.ic_code} config...")
    if args.ic_code == "music":
        cmd = py_cli("icpipe.cli.make_music_conf",
                     "-N", args.ngrid, "-z", args.zstart, "-L", args.lbox,
                     "--fix-amplitude", args.fix_amplitude)
        if args.seed:
            cmd += ["--seed", str(args.seed)]
    else:
        cmd = py_cli("icpipe.cli.make_monofonic_conf",
                     "-N", args.ngrid, "-z", args.zstart, "-L", args.lbox,
                     "--fixing", args.fix_amplitude,
                     "--lpt-order", args.lpt_order,
                     "--nthreads", args.nthreads)
        if args.seed:
            cmd += ["--seed", str(args.seed)]
    run(cmd)
    return conf


def step_run_ic(args, stem, conf):
    ic_file = f"data/ics_swift_{stem}.hdf5"
    if os.path.exists(os.path.join(REPO, ic_file)):
        log(f"IC file already exists — skipping ({ic_file})")
        return ic_file
    log(f"Running {args.ic_code}...")
    run([ic_binary(args.ic_code), conf])
    mib = peak_child_mib()

    # Both codes write a CLASS ini to the CWD; normalize it to one path so the
    # CLASS step is code-agnostic.
    class_ini = f"conf/input_class_parameters_{stem}.ini"

    if args.ic_code == "music":
        # MUSIC writes input_class_parameters.ini and wnoise_*.bin to the CWD.
        move_if("input_class_parameters.ini", class_ini)
        for f in sorted(glob(os.path.join(REPO, "wnoise_*.bin"))):
            move_if(os.path.basename(f), f"data/{os.path.basename(f)}")
    else:
        # monofonIC prefixes most aux outputs with the config basename (conf file
        # minus dir/ext) and writes them to the CWD (relative_to_config = no).
        base = os.path.splitext(os.path.basename(conf))[0]  # CV_22_monofonIC_{stem}
        move_if(f"{base}_input_class_parameters.ini", class_ini)
        for name in ("input_powerspec.txt", "input_transfer.txt", "log.txt"):
            move_if(f"{base}_{name}", f"data/{stem}_{name}")
        for f in sorted(glob(os.path.join(REPO, f"{base}_input_powerspec_sampled*.txt"))):
            suffix = os.path.basename(f)[len(base) + 1:]
            move_if(os.path.basename(f), f"data/{stem}_{suffix}")
        # GrowthFactors.txt is written with a fixed name (not config-prefixed).
        move_if("GrowthFactors.txt", f"data/{stem}_GrowthFactors.txt")

    print(f"    Saved: {ic_file}")
    print(f"    Peak child memory: {mib:.1f} MiB")
    return ic_file


def find_class_bin():
    """Locate a standalone CLASS binary. Built by MUSIC's CMake; monofonIC links
    CLASS as a library and builds no standalone binary, so a monofonic-only setup
    relies on a CLASS binary (or a prebuilt data/class_pk_z0_pk.dat) from MUSIC."""
    for p in ("music_build/_deps/class-build/class",
              "monofonic_build/_deps/class-build/class"):
        if os.path.exists(os.path.join(REPO, p)):
            return p
    return None


def step_class(args, stem):
    if os.path.exists(os.path.join(REPO, CLASS_PKS)):
        log(f"CLASS P(k) already exists — skipping ({CLASS_PKS})")
        return CLASS_PKS

    # Both codes write a CLASS ini, normalized to this path in step_run_ic.
    class_ini = f"conf/input_class_parameters_{stem}.ini"
    class_bin = find_class_bin()
    if not os.path.exists(os.path.join(REPO, class_ini)) or class_bin is None:
        reason = ("no CLASS ini found" if not os.path.exists(os.path.join(REPO, class_ini))
                  else "no standalone CLASS binary (build MUSIC once to get one)")
        log(f"Cannot build {CLASS_PKS}: {reason}. Theory overlay will be skipped.")
        return None

    log(f"Running CLASS at z=0 (back-scaled at plot time to z={args.zstart})...")
    fd, tmp = tempfile.mkstemp(suffix=".ini")
    os.close(fd)
    with open(os.path.join(REPO, class_ini)) as fin:
        lines = []
        for line in fin:
            s = line.strip()
            if s.startswith("output ="):
                line = "output = mPk\n"
            elif s.startswith("z_pk ="):
                line = "z_pk = 0\n"
            elif s.startswith("extra metric transfer functions") or s.startswith("gauge"):
                continue
            lines.append(line)
        lines.append("root = class_pk_z0_\n")
    with open(tmp, "w") as fout:
        fout.writelines(lines)
    run([class_bin, tmp])
    shutil.move(os.path.join(REPO, "class_pk_z0_pk.dat"), os.path.join(REPO, CLASS_PKS))
    os.remove(tmp)
    print(f"    Saved: {CLASS_PKS}")
    return CLASS_PKS


def step_rbins(args, stem, ic_file):
    rbins = f"data/rbins_{stem}.txt"
    if os.path.exists(os.path.join(REPO, rbins)):
        log(f"rbins already exist — skipping ({rbins})")
        return rbins
    log("Generating rbins...")
    run(py_cli("icpipe.cli.make_rbins", "--hdf5", ic_file))
    return rbins


def step_xi(args, stem, ic_file, rbins):
    xi_file = f"data/xi_{stem}.txt"
    if float(args.zstart) > 10:
        log(f"Skipping Corrfunc xi(r) — z={args.zstart} > 10 (shot-noise dominated; use CIC)")
        return
    log(f"Measuring xi(r) with compute_xi ({args.nthreads} threads)...")
    with open(os.path.join(REPO, xi_file), "w") as out:
        run(["./bin/compute_xi", ic_file, rbins, str(args.nthreads)], stdout=out)
    print(f"    Saved: {xi_file}")


def step_xi_cic(args, stem, ic_file):
    xi_cic = f"data/xi_cic_{stem}.txt"
    log(f"Measuring xi(r) and psi(r) on CIC grid ({args.nthreads} threads)...")
    run(["./bin/compute_xi_cic", "--input", ic_file, "--Ngrid", "128",
         "--nthreads", str(args.nthreads), "--output", xi_cic, "--vel"])
    print(f"    Saved: {xi_cic}")
    print(f"    Saved: data/vel_cic_{stem}.txt")


def step_pk(args, stem, ic_file):
    pk_file = f"data/pk_{stem}.txt"
    log("Measuring P(k) with compute_pk...")
    run(py_cli("icpipe.cli.compute_pk", ic_file, "--H0", H0, "-o", pk_file))
    print(f"    Saved: {pk_file}")
    return pk_file


def step_plot(args, stem, pk_file, theory):
    plot_file = f"plots/pk_{stem}.png"
    log("Plotting diagnostics with plot_ic...")
    cmd = py_cli("icpipe.cli.plot_ic", pk_file,
                 "--H0", H0, "--Omega_m", OMEGA_M, "--Omega_b", OMEGA_B,
                 "-o", plot_file)
    if theory:
        cmd += ["--theory", theory, "--theory-zref", "0"]
    run(cmd)
    return plot_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="End-to-end IC generation and validation (MUSIC or monofonIC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ic-code", choices=["music", "monofonic"], default="music",
                   help="IC generator")
    p.add_argument("--ngrid", default="256", help="Particles per side (N)")
    p.add_argument("--lbox", default="1000", help="Box size in Mpc/h")
    p.add_argument("--zstart", default="200", help="Starting redshift")
    p.add_argument("--nthreads",
                   default=str(os.cpu_count() or 4),
                   help="OpenMP threads")
    p.add_argument("--fix-amplitude", choices=["yes", "no"], default="yes",
                   help="Fix Fourier mode amplitudes (MUSIC fix_mode_amplitude / monofonic DoFixing)")
    p.add_argument("--lpt-order", choices=["1", "2", "3"], default="3",
                   help="LPT order (monofonic only; MUSIC uses use_2LPT)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for the IC white noise (paired-suite runs)")
    p.add_argument("--ic-source-dir", default=None,
                   help="Path to the IC-code source checkout (else sibling dir or clone)")
    p.add_argument("--corrfunc-dir", default=None, help="Path to a built Corrfunc tree")
    p.add_argument("--stop-after-ic", action="store_true",
                   help="Exit after Step 4 (IC generation), skipping CLASS / xi / P(k) / plots")
    args = p.parse_args()

    os.makedirs(os.path.join(REPO, "data"), exist_ok=True)
    os.makedirs(os.path.join(REPO, "plots"), exist_ok=True)
    os.makedirs(os.path.join(REPO, "conf"), exist_ok=True)

    stem = stem_for(args)

    step_build_ic(args, stem)
    step_build_compute_xi(args)
    conf = step_gen_config(args, stem)
    ic_file = step_run_ic(args, stem, conf)

    if args.stop_after_ic:
        log("--stop-after-ic set: exiting before CLASS P(k).")
        return

    theory = step_class(args, stem)
    rbins = step_rbins(args, stem, ic_file)
    step_xi(args, stem, ic_file, rbins)
    step_xi_cic(args, stem, ic_file)
    pk_file = step_pk(args, stem, ic_file)
    plot_file = step_plot(args, stem, pk_file, theory)

    log("Pipeline complete.")
    print(f"    IC code : {args.ic_code}")
    print(f"    IC file : {ic_file}")
    print(f"    P(k)    : {pk_file}")
    print(f"    Plot    : {plot_file}")


if __name__ == "__main__":
    main()
