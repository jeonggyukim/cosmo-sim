#!/usr/bin/env bash
# run_pipeline.sh — End-to-end IC generation and validation pipeline.
#
# Steps (each skipped if output already exists):
#   1. Build MUSIC2         — compile the MUSIC2 binary (clones from GitHub if needed)
#   2. Build compute_xi     — compile the Corrfunc-based ξ(r) estimator
#   3. Generate MUSIC2 conf — expand the template conf for the chosen NGRID, ZSTART, LBOX
#   4. Run MUSIC2           — generate IC HDF5 + input_class_parameters.ini
#   5. Run CLASS P(k)       — adapt MUSIC2's CLASS ini for P(k) output; run CLASS
#   6. Generate rbins       — bin edges for compute_xi (rmin=2Δx, rmax=L/3, in Mpc)
#   7. Measure ξ(r)         — pair counts via Corrfunc (low-z only; shot-noise dominated at z≳10)
#   8. Measure ξ(r), ψ(r)  — CIC grid estimator; works at any z; includes velocity correlation
#   9. Measure P(k)         — CIC + FFT estimator, overlaid with CLASS theory
#
# Usage:
#   ./run_pipeline.sh [--ngrid NGRID] [--lbox LBOX] [--zstart ZSTART] [--nthreads NTHREADS]
#
#   --ngrid          particles per side (default: 256)
#   --lbox           box side length in Mpc/h (default: 687)
#   --zstart         IC starting redshift (default: 2)
#   --nthreads       OpenMP threads for compute_xi (default: 8)
#   --fix-amplitude  yes|no — fix Fourier mode amplitudes to sqrt(P(k)) (default: yes)
#                    When no, '_nofix' is appended to all output stems.
#
# Examples:
#   ./run_pipeline.sh
#   ./run_pipeline.sh --ngrid 512 --lbox 500 --zstart 127
#
# All outputs are keyed by STEM = n{NGRID}_z{ZSTART}_L{LBOX}:
#   conf/CV_22_MUSIC_{STEM}.conf           — MUSIC2 config
#   conf/input_class_parameters_{STEM}.ini — CLASS ini written by MUSIC2 (transfer functions)
#   data/ics_swift_{STEM}.hdf5             — IC particle file (SWIFT format, coords in Mpc)
#   data/class_pk_z{ZSTART}_pk.dat         — CLASS matter P(k) at z=ZSTART
#   data/rbins_{STEM}.txt                  — Corrfunc radial bin edges (Mpc)
#   data/xi_{STEM}.txt                     — measured ξ(r) (Corrfunc)
#   data/xi_cic_{STEM}.txt                 — measured ξ(r) (CIC grid)
#   data/vel_cic_{STEM}.txt                — measured ψ(r) (CIC grid velocity)
#   data/pk_{STEM}.txt                     — measured P(k) table
#   plots/pk_{STEM}.png                    — P(k) + ξ(r) validation figure

set -euo pipefail

# Always run from the repo root regardless of where the script is invoked from
cd "$(dirname "$(readlink -f "$0")")"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
NGRID=256
LBOX=1024
ZSTART=200
# Default to all logical CPUs; detect cross-platform (macOS: sysctl, Linux: nproc)
NTHREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)
FIX_AMP=yes   # fix_mode_amplitude: yes = fixed amplitudes (CV); no = Gaussian draw
SEED=""       # random seed override for the IC level; empty = use template default
MUSIC2_DIR=""     # optional: path to MUSIC2 source (default: ~/Dropbox/Projects/MUSIC2 or cloned)
CORRFUNC_DIR=""   # optional: path to built Corrfunc (default: ~/Corrfunc or cloned)

while [[ $# -gt 0 ]]; do
    case $1 in
        --ngrid)         NGRID="$2";        shift 2 ;;
        --lbox)          LBOX="$2";         shift 2 ;;
        --zstart)        ZSTART="$2";       shift 2 ;;
        --nthreads)      NTHREADS="$2";     shift 2 ;;
        --fix-amplitude) FIX_AMP="$2";      shift 2 ;;
        --seed)          SEED="$2";         shift 2 ;;
        --music2-dir)    MUSIC2_DIR="$2";   shift 2 ;;
        --corrfunc-dir)  CORRFUNC_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"
           echo "Usage: $0 [--ngrid N] [--lbox L] [--zstart Z] [--nthreads T]"
           echo "          [--fix-amplitude yes|no] [--seed INT]"
           echo "          [--music2-dir /path/to/MUSIC2]"
           echo "          [--corrfunc-dir /path/to/Corrfunc]"
           exit 1 ;;
    esac
done

# CV_22 cosmology (Illustris/TNG) — must match conf/CV_22_MUSIC_template.conf
H0=67.11
OMEGA_M=0.3
OMEGA_B=0.049

# ---------------------------------------------------------------------------
# Derived paths — all keyed by STEM so multiple runs coexist without collision
# ---------------------------------------------------------------------------
AMP_SUFFIX=$([ "$FIX_AMP" = "no" ] && echo "_nofix" || echo "")
SEED_SUFFIX=$([ -n "$SEED" ] && echo "_s${SEED}" || echo "")
STEM="n${NGRID}_z${ZSTART}_L${LBOX}${AMP_SUFFIX}${SEED_SUFFIX}"
IC_FILE="data/ics_swift_${STEM}.hdf5"
CONF_FILE="conf/CV_22_MUSIC_${STEM}.conf"
# input_class_parameters.ini is written by MUSIC2 to CWD during the IC run;
# we move it to conf/ immediately after so it stays with the other run artifacts.
CLASS_INI="conf/input_class_parameters_${STEM}.ini"
CLASS_PKS="data/class_pk_z0_pk.dat"
RBINS_FILE="data/rbins_${STEM}.txt"
MUSIC_BIN="music_build/MUSIC"
CLASS_BIN="music_build/_deps/class-build/class"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo; echo "==> $*"; }

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
mkdir -p data plots conf

# ---------------------------------------------------------------------------
# Step 1: Build MUSIC2
# Skipped if the binary already exists.
# build-music.sh clones MUSIC2 from GitHub if the source directory is missing,
# then compiles with CMake (also builds CLASS as a dependency).
# ---------------------------------------------------------------------------
if [ ! -f "$MUSIC_BIN" ]; then
    log "Building MUSIC2..."
    if [ -n "$MUSIC2_DIR" ]; then
        MUSIC2_SOURCE_DIR="$MUSIC2_DIR" ./build-music.sh
    else
        ./build-music.sh
    fi
else
    log "MUSIC2 already built — skipping"
fi

# ---------------------------------------------------------------------------
# Step 2: Build compute_xi
# Skipped if the binary already exists.
# Compiled via the repo-root Makefile; links against Corrfunc.
# If --corrfunc-dir is given, pass CORRFUNCDIR to make.
# ---------------------------------------------------------------------------
if [ ! -f "compute_xi" ]; then
    log "Building compute_xi..."
    if [ -n "$CORRFUNC_DIR" ]; then
        make CORRFUNCDIR="$CORRFUNC_DIR"
    else
        make
    fi
else
    log "compute_xi already built — skipping"
fi

# ---------------------------------------------------------------------------
# Step 3: Generate MUSIC2 config
# Skipped if the conf file already exists.
# make_music_conf.py expands conf/CV_22_MUSIC_template.conf with the chosen
# N, z, L and writes conf/CV_22_MUSIC_{STEM}.conf.
# ---------------------------------------------------------------------------
if [ ! -f "$CONF_FILE" ]; then
    log "Generating MUSIC2 config..."
    conda run -n cosmo python scripts/make_music_conf.py -N "$NGRID" -z "$ZSTART" -L "$LBOX" --fix-amplitude "$FIX_AMP" ${SEED:+--seed "$SEED"}
else
    log "MUSIC2 config already exists — skipping ($CONF_FILE)"
fi

# ---------------------------------------------------------------------------
# Step 4: Run MUSIC2
# Skipped if the IC HDF5 already exists.
# MUSIC2 generates the IC particle file (SWIFT HDF5 format, coordinates in Mpc)
# and writes input_class_parameters.ini to CWD with the cosmological parameters
# it passed to CLASS for the transfer function calculation.
# We immediately move that file to conf/ so it lives alongside the other
# run-specific config files and is not left cluttering the repo root.
# ---------------------------------------------------------------------------
if [ ! -f "$IC_FILE" ]; then
    log "Running MUSIC2..."
    "$MUSIC_BIN" "$CONF_FILE"
    mv input_class_parameters.ini "$CLASS_INI"
    # MUSIC2 always writes wnoise_NNNN.bin to CWD (path is hardcoded); move to data/
    for f in wnoise_*.bin; do [ -f "$f" ] && mv "$f" "data/$f" && echo "    Saved: data/$f"; done
    echo "    Saved: $IC_FILE"
    echo "    Saved: $CLASS_INI"
else
    log "IC file already exists — skipping ($IC_FILE)"
fi

# ---------------------------------------------------------------------------
# Step 5: Generate CLASS matter P(k) at z=Z
# Skipped if the output file already exists.
# input_class_parameters.ini (written by MUSIC2) is configured for transfer
# function output (dTk, vTk) in synchronous gauge. We adapt it for matter P(k)
# by changing the output type to mPk, removing transfer-function-specific
# settings (extra metric transfer functions, gauge), and appending z_pk and
# root. A temp file with a .ini extension is required — CLASS ignores files
# without a recognised extension.
# CLASS writes class_pk_z{Z}_pk.dat to CWD; we move it to data/.
# ---------------------------------------------------------------------------
if [ ! -f "$CLASS_PKS" ]; then
    # CLASS P(k) is generated at z=0, and plot_ic.py back-scales it to the
    # IC redshift using the radiation-free growth factor D_+^no-rad(z) that
    # matches MUSIC2's ZeroRadiation=true convention.  See notes/cosmo_ic.tex
    # §"Radiation and the back-scaling growth factor".
    log "Running CLASS at z=0 (back-scaled at plot time to z=${ZSTART})..."
    if [ ! -f "$CLASS_INI" ]; then
        echo "Error: $CLASS_INI not found — was MUSIC2 run with transfer_function = CLASS?"
        exit 1
    fi
    # mktemp on macOS requires X's at the end, so rename to add .ini suffix
    TMP_INI=$(mktemp /tmp/class_pk_XXXXXX) && mv "$TMP_INI" "${TMP_INI}.ini" && TMP_INI="${TMP_INI}.ini"
    sed \
        -e "s/^output =.*/output = mPk/" \
        -e "s/^z_pk =.*/z_pk = 0/" \
        -e "/^extra metric transfer functions/d" \
        -e "/^gauge/d" \
        "$CLASS_INI" > "$TMP_INI"
    echo "root = class_pk_z0_" >> "$TMP_INI"
    "$CLASS_BIN" "$TMP_INI"
    mv "class_pk_z0_pk.dat" "$CLASS_PKS"
    rm "$TMP_INI"
    echo "    Saved: $CLASS_PKS"
else
    log "CLASS P(k) already exists — skipping ($CLASS_PKS)"
fi

# ---------------------------------------------------------------------------
# Step 6: Generate Corrfunc radial bin file
# Skipped if the rbins file already exists.
# make_rbins.py reads the HDF5 header to get BoxSize and N, then sets
# rmin = 2 × mean particle spacing, rmax = L/3, logarithmically spaced.
# Bins are in Mpc (matching SWIFT's coordinate units).
# ---------------------------------------------------------------------------
if [ ! -f "$RBINS_FILE" ]; then
    log "Generating rbins..."
    conda run -n cosmo python scripts/make_rbins.py --hdf5 "$IC_FILE"
else
    log "rbins already exist — skipping ($RBINS_FILE)"
fi

# ---------------------------------------------------------------------------
# Step 7: Measure ξ(r) with compute_xi
# Skipped at high z (z > 10): at high redshift the linear power spectrum is
# suppressed by D²(z) ≪ 1, so P_signal(k) ≪ P_shot = V/N for nearly all k,
# and Corrfunc pair-counting ξ(r) is completely shot-noise dominated.
# The CIC estimator (step 8) is preferred at any z.
# ---------------------------------------------------------------------------
XI_FILE="data/xi_${STEM}.txt"
if (( $(echo "$ZSTART > 10" | bc -l) )); then
    log "Skipping Corrfunc ξ(r) — z=${ZSTART} > 10 (shot-noise dominated; use CIC P(k))"
else
    log "Measuring ξ(r) with compute_xi (${NTHREADS} threads)..."
    ./compute_xi "$IC_FILE" "$RBINS_FILE" "$NTHREADS" > "$XI_FILE"
    echo "    Saved: $XI_FILE"
fi

# ---------------------------------------------------------------------------
# Step 8: Measure ξ(r) and ψ(r) on a CIC grid with compute_xi_cic
# Not skipped — always re-runs.
# Assigns particle positions and velocities to a 128³ CIC grid, then
# computes the density autocorrelation ξ(r) and the velocity autocorrelation
# ψ(r) = ⟨v_pec(x)·v_pec(x+r)⟩ via lag-sum over the grid.
# Works at any redshift (avoids Corrfunc shot-noise limitations at high z).
# Outputs:
#   data/xi_cic_{STEM}.txt  — CIC density correlation
#   data/vel_cic_{STEM}.txt — CIC velocity correlation
# ---------------------------------------------------------------------------
XI_CIC_FILE="data/xi_cic_${STEM}.txt"
VEL_CIC_FILE="data/vel_cic_${STEM}.txt"
log "Measuring ξ(r) and ψ(r) on CIC grid with compute_xi_cic (${NTHREADS} threads)..."
./compute_xi_cic \
    --input    "$IC_FILE" \
    --Ngrid    128 \
    --nthreads "$NTHREADS" \
    --output   "$XI_CIC_FILE" \
    --vel
echo "    Saved: $XI_CIC_FILE"
echo "    Saved: $VEL_CIC_FILE"

# ---------------------------------------------------------------------------
# Step 10: Measure P(k) with compute_pk.py
# Not skipped — always re-runs.
# CIC mass assignment on an Ngrid³ mesh, FFT, CIC window deconvolution,
# shot-noise subtraction (P_shot = V/N), binning into log k-shells.
# Outputs an ASCII table: data/pk_{STEM}.txt
# ---------------------------------------------------------------------------
PK_FILE="data/pk_${STEM}.txt"
log "Measuring P(k) with compute_pk.py..."
conda run -n cosmo python scripts/compute_pk.py \
    "$IC_FILE" \
    --H0 "$H0" \
    -o "$PK_FILE"
echo "    Saved: $PK_FILE"

# ---------------------------------------------------------------------------
# Step 11: Plot diagnostics with plot_ic.py
# Reads pk_{STEM}.txt; auto-detects xi_{STEM}.txt, xi_cic_{STEM}.txt,
# vel_cic_{STEM}.txt alongside it; overlays CLASS theory curves.
#
# MUSIC2 uses ZeroRadiation=true, which back-scales from z=0 with the
# matter-only growth factor D_+^no-rad(z).  The reason is that downstream
# N-body/hydro codes (SWIFT, GADGET, ...) ignore radiation in their
# background Friedmann evolution, so the ICs must be generated with the
# same Omega_r=0 growth history for consistency — otherwise the IC
# amplitude at z_start would disagree with what the N-body integrator
# expects a linear mode to be at that redshift.  Boltzmann codes (CLASS,
# CAMB) include radiation, so their D_+(z) differs by ~11% at z=200,
# producing a ~5-10% P(k) suppression if compared without back-scaling.
# For a clean apples-to-apples comparison at any z_start, we always feed
# plot_ic.py the CLASS P(k) at z=0 and let it back-scale with the same
# D_+^no-rad convention MUSIC2 uses internally.
# See notes/cosmo_ic.tex §"Radiation and the back-scaling growth factor".
#
# Outputs: plots/pk_{STEM}.png
# ---------------------------------------------------------------------------
PLOT_FILE="plots/pk_${STEM}.png"
log "Plotting diagnostics with plot_ic.py..."
conda run -n cosmo python scripts/plot_ic.py \
    "$PK_FILE" \
    --theory "$CLASS_PKS" \
    --theory-zref 0 \
    --H0 "$H0" \
    --Omega_m "$OMEGA_M" \
    --Omega_b "$OMEGA_B" \
    -o "$PLOT_FILE"

log "Pipeline complete."
echo "    IC file : $IC_FILE"
echo "    P(k)    : $PK_FILE"
echo "    Plot    : $PLOT_FILE"
