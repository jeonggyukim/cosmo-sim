#!/usr/bin/env bash
# run_pipeline.sh — End-to-end IC generation and validation pipeline.
#
# Steps (each skipped if output already exists):
#   1. Build MUSIC2
#   2. Build compute_xi
#   3. Generate CLASS P(k) at zstart
#   4. Generate MUSIC2 config
#   5. Run MUSIC2 → IC HDF5
#   6. Generate rbins for compute_xi
#   7. Measure ξ(r) with compute_xi
#   8. Measure P(k) with compute_pk.py
#
# Usage:
#   ./run_pipeline.sh                     # defaults: N=256, L=1000, Z=45
#   N=512 L=500 Z=127 ./run_pipeline.sh   # override via env vars

set -euo pipefail

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N=${N:-256}
L=${L:-1000}
Z=${Z:-45}
NTHREADS=${NTHREADS:-8}
H0=67.11
OMEGA_M=0.3
OMEGA_B=0.049

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
STEM="n${N}_z${Z}_L${L}"
IC_FILE="data/ics_swift_${STEM}.hdf5"
CONF_FILE="conf/CV_22_MUSIC_${STEM}.conf"
CLASS_INI="data/class_pk.ini"
CLASS_PKS="data/class_pk_z${Z}_pk.dat"
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
mkdir -p data plots

# ---------------------------------------------------------------------------
# Step 1: Build MUSIC2
# ---------------------------------------------------------------------------
if [ ! -f "$MUSIC_BIN" ]; then
    log "Building MUSIC2..."
    ./build-music.sh
else
    log "MUSIC2 already built — skipping"
fi

# ---------------------------------------------------------------------------
# Step 2: Build compute_xi
# ---------------------------------------------------------------------------
if [ ! -f "compute_xi" ]; then
    log "Building compute_xi..."
    make
else
    log "compute_xi already built — skipping"
fi

# ---------------------------------------------------------------------------
# Step 3: Generate CLASS P(k)
# ---------------------------------------------------------------------------
if [ ! -f "$CLASS_PKS" ]; then
    log "Running CLASS for z=${Z}..."
    TMP_INI=$(mktemp /tmp/class_pk_XXXXXX.ini)
    sed \
        -e "s/^z_pk =.*/z_pk = ${Z}/" \
        -e "s/^root =.*/root = class_pk_z${Z}_/" \
        "$CLASS_INI" > "$TMP_INI"
    "$CLASS_BIN" "$TMP_INI"
    mv "class_pk_z${Z}_pk.dat" "$CLASS_PKS"
    rm "$TMP_INI"
    echo "    Saved: $CLASS_PKS"
else
    log "CLASS P(k) already exists — skipping ($CLASS_PKS)"
fi

# ---------------------------------------------------------------------------
# Step 4: Generate MUSIC2 config
# ---------------------------------------------------------------------------
if [ ! -f "$CONF_FILE" ]; then
    log "Generating MUSIC2 config..."
    conda run -n cosmo python scripts/make_music_conf.py -N "$N" -z "$Z" -L "$L"
else
    log "MUSIC2 config already exists — skipping ($CONF_FILE)"
fi

# ---------------------------------------------------------------------------
# Step 5: Run MUSIC2
# ---------------------------------------------------------------------------
if [ ! -f "$IC_FILE" ]; then
    log "Running MUSIC2..."
    "$MUSIC_BIN" "$CONF_FILE"
    echo "    Saved: $IC_FILE"
else
    log "IC file already exists — skipping ($IC_FILE)"
fi

# ---------------------------------------------------------------------------
# Step 6: Generate rbins
# ---------------------------------------------------------------------------
if [ ! -f "$RBINS_FILE" ]; then
    log "Generating rbins..."
    conda run -n cosmo python scripts/make_rbins.py --hdf5 "$IC_FILE"
else
    log "rbins already exist — skipping ($RBINS_FILE)"
fi

# ---------------------------------------------------------------------------
# Step 7: Measure ξ(r)
# ---------------------------------------------------------------------------
log "Measuring ξ(r) with compute_xi (${NTHREADS} threads)..."
./compute_xi "$IC_FILE" "$RBINS_FILE" "$NTHREADS"

# ---------------------------------------------------------------------------
# Step 8: Measure P(k)
# ---------------------------------------------------------------------------
log "Measuring P(k) with compute_pk.py..."
conda run -n cosmo python scripts/compute_pk.py \
    "$IC_FILE" \
    --theory "$CLASS_PKS" \
    --H0 "$H0" \
    --Omega_m "$OMEGA_M" \
    --Omega_b "$OMEGA_B" \
    -o "plots/pk_${STEM}.png"

log "Pipeline complete."
echo "    IC file : $IC_FILE"
echo "    P(k)    : plots/pk_${STEM}.png"
