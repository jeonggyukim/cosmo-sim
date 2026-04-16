#!/usr/bin/env bash
# clean.sh — Remove generated files from the pipeline.
#
# Usage:
#   ./clean.sh           # remove fast-to-regenerate outputs (plots, tables, rbins, configs)
#   ./clean.sh --all     # also remove ICs, CLASS P(k), wnoise (slow to regenerate)

set -euo pipefail

ALL=false
if [[ "${1:-}" == "--all" ]]; then
    ALL=true
fi

log() { echo "==> $*"; }

# ---------------------------------------------------------------------------
# Always remove: plots, measured P(k)/xi tables, rbins, generated configs
# ---------------------------------------------------------------------------
log "Removing plots..."
rm -f plots/*.png plots/*.pdf

log "Removing measured P(k) and xi tables..."
rm -f data/pk_*.txt data/xi_*.txt data/vel_cic_*.txt

log "Removing rbins files..."
rm -f data/rbins*.txt

log "Removing generated MUSIC2 configs and CLASS ini files..."
rm -f conf/CV_22_MUSIC_n*.conf
rm -f conf/input_class_parameters_*.ini

log "Removing compiled compute_xi binary..."
rm -f compute_xi

log "Removing CLASS P(k) outputs..."
rm -f data/class_pk_z*.dat

if $ALL; then
    # -----------------------------------------------------------------------
    # Also remove: IC HDF5 files, wnoise binaries (slow to regenerate)
    # -----------------------------------------------------------------------
    log "Removing IC HDF5 files..."
    rm -f data/ics_swift_*.hdf5

    log "Removing wnoise binary files..."
    rm -f data/wnoise_*.bin

    log "Removing MUSIC2 build..."
    rm -rf music_build/

    echo
    echo "Full clean done. Re-run ./run_pipeline.sh to regenerate everything."
else
    echo
    echo "Clean done. Re-run ./run_pipeline.sh to regenerate outputs."
    echo "Use './clean.sh --all' to also remove ICs, CLASS outputs, and MUSIC2 build."
fi
