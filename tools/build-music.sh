#!/usr/bin/env bash
#
# build-music.sh — Build MUSIC2 from source into music_build/
#
# Usage:
#   ./build-music.sh                          # auto-detect or clone MUSIC2 source
#   ./build-music.sh -d DIR                   # builds into DIR
#   MUSIC2_SOURCE_DIR=/path ./build-music.sh  # use specific MUSIC2 source directory
#
# MUSIC2 source directory resolution order:
#   1. $MUSIC2_SOURCE_DIR environment variable (if set)
#   2. $HOME/Dropbox/Projects/MUSIC2          (macOS default)
#   3. $HOME/MUSIC2                           (cluster default)
#   4. Clone from https://github.com/cosmo-sims/MUSIC2 into option 3
#
# Skips compilation if the MUSIC binary already exists in BUILDDIR.

# --- Locate MUSIC2 source ---
# Resolution order:
#   1. $MUSIC2_SOURCE_DIR env var (set by run-pipeline --ic-source-dir)
#   2. ../MUSIC2  relative to this repo (sibling directory, default)
#   3. Clone from GitHub into ../MUSIC2 if not found
REPO_ROOT="$(dirname "$(readlink -f "$0")")"
DEFAULT_MUSICDIR="$(readlink -f "$REPO_ROOT/../MUSIC2")"

if [ -n "${MUSIC2_SOURCE_DIR:-}" ]; then
    MUSICDIR="$MUSIC2_SOURCE_DIR"
else
    MUSICDIR="$DEFAULT_MUSICDIR"
fi

# --- Clone MUSIC2 if source directory does not exist ---
if [ ! -d "$MUSICDIR" ]; then
    echo "MUSIC2 source not found at $MUSICDIR — cloning from GitHub..."
    git clone https://github.com/cosmo-sims/MUSIC2.git "$MUSICDIR"
fi

# --- Parse command line arguments ---
BUILDDIR_DEFAULT="$(dirname "$(readlink -f "$0")")/music_build"
BUILDDIR=""

while getopts "d:" opt; do
    case $opt in
        d)
            BUILDDIR="$OPTARG"
            ;;
        \?)
            echo "Usage: $0 [-d BUILDDIR]"
            exit 1
            ;;
    esac
done

# Fall back to default build directory if not specified
BUILDDIR="${BUILDDIR:-$BUILDDIR_DEFAULT}"

# --- Build ---
if [ -d "$BUILDDIR" ] && [ -f "$BUILDDIR/MUSIC" ]; then
    echo "MUSIC binary already exists in $BUILDDIR, skipping compilation"
else
    # Set up compilers / libraries
    if [[ $(uname -s) == "Darwin" ]]; then
        # macOS: use Homebrew gfortran; fftw/gsl/hdf5 found via CMake automatically
        export FC=gfortran-14
    else
        # Cluster: load required modules
        module purge 1>/dev/null 2>&1
        module load gnu12/12.2.0
        module load openmpi4/4.1.5
        module load fftw/3.3.10
        module load hdf5/1.14.0
        module load gsl/2.7.1
        module load cmake/4.0.0

        export CMAKE_LIBRARY_PATH=$CMAKE_LIBRARY_PATH:$FFTW_LIB:$GSL_LIB
        export CMAKE_INCLUDE_PATH=$CMAKE_INCLUDE_PATH:$FFTW_INC:$GSL_INC
    fi

    # Create build directory if it doesn't exist
    if [ ! -d "$BUILDDIR" ]; then
        if [ -e "$BUILDDIR" ]; then
            echo "Error: $BUILDDIR exists but is not a directory"
            exit 1
        fi
        echo "Creating build directory: $BUILDDIR"
        mkdir -p "$BUILDDIR" || { echo "Error: Failed to create $BUILDDIR"; exit 1; }
    fi

    cd "$BUILDDIR" || { echo "Error: Failed to cd to $BUILDDIR"; exit 1; }
    echo "Compiling MUSIC in $BUILDDIR..."
    cmake $MUSICDIR
    make -j
fi
