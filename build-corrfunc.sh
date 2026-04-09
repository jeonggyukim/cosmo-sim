#!/usr/bin/env bash
#
# prepare-corrfunc.sh — Clone and build Corrfunc from source
#
# Usage:
#   ./prepare-corrfunc.sh
#
# Skips compilation if the Corrfunc library already exists.
#
# Platform behaviour:
#   macOS (Darwin, user jgkim): CORRFUNCDIR set to Dropbox path
#   Cluster (non-Darwin):       CORRFUNCDIR=$HOME/Corrfunc

# --- Locate / clone Corrfunc source ---
if [[ $(uname -s) == "Darwin" && $(whoami) == "jgkim" ]]; then
    CORRFUNCDIR=$HOME/Dropbox/Projects/Corrfunc
else
    CORRFUNCDIR=$HOME/Corrfunc
fi

CORRFUNC_REPO="https://github.com/manodeep/Corrfunc.git"

if [ ! -d "$CORRFUNCDIR" ]; then
    echo "Cloning Corrfunc into $CORRFUNCDIR..."
    git clone "$CORRFUNC_REPO" "$CORRFUNCDIR" || { echo "Error: git clone failed"; exit 1; }
fi

# --- Build ---
LIB="$CORRFUNCDIR/theory/xi/libcountpairs_xi.a"

if [ -f "$LIB" ]; then
    echo "Corrfunc already built at $CORRFUNCDIR, skipping compilation"
else
    echo "Building Corrfunc in $CORRFUNCDIR..."
    make -C "$CORRFUNCDIR" -j || { echo "Error: make failed"; exit 1; }
    echo "Done."
fi
