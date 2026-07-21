#!/usr/bin/env bash
#
# build-monofonic.sh — Build MUSIC2-monofonIC from source into monofonic_build/
#
# Usage:
#   ./build-monofonic.sh                              # auto-detect or clone monofonIC source
#   ./build-monofonic.sh -d DIR                       # builds into DIR
#   MONOFONIC_SOURCE_DIR=/path ./build-monofonic.sh   # use specific monofonIC source directory
#
# monofonIC source directory resolution order:
#   1. $MONOFONIC_SOURCE_DIR environment variable (if set)
#   2. ../monofonIC  relative to this repo (sibling directory, default)
#   3. Clone from https://github.com/cosmo-sims/monofonIC into option 2
#
# CLASS is pulled in automatically by monofonIC's CMake (FetchContent), so there
# is no separate CLASS build step here.
#
# Skips compilation if the monofonIC binary already exists in BUILDDIR.
#
# Build options (differences from a stock monofonIC build):
#   -DENABLE_PLT=ON         PLT (particle linear theory) correction — monofonIC
#                           defaults this OFF, but PLT is the main reason this
#                           pipeline uses monofonIC over legacy MUSIC.
#   -DENABLE_MPI=ON         monofonIC's source (e.g. grid_ghosts.hh) does not
#                           compile without MPI, so MPI is required. Needs
#                           open-mpi + FFTW3-MPI + parallel HDF5. The binary still
#                           runs single-rank (no mpirun needed).
#   -DENABLE_PANPHASIA=OFF  drop the Fortran PANPHASIA generator; the pipeline
#                           uses the NGENIC generator.
#   -DENABLE_CLASS=ON       CLASS transfer-function module (monofonIC default).

# --- Locate monofonIC source ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"          # <repo>/tools
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"            # <repo>
PROJECTS_DIR="$(cd "$REPO_ROOT/.." && pwd)"          # parent of <repo>
DEFAULT_MONOFONICDIR="$PROJECTS_DIR/monofonIC"       # sibling of the repo root

if [ -n "${MONOFONIC_SOURCE_DIR:-}" ]; then
    MONOFONICDIR="$MONOFONIC_SOURCE_DIR"
else
    MONOFONICDIR="$DEFAULT_MONOFONICDIR"
fi

# --- Clone monofonIC if source directory does not exist ---
if [ ! -d "$MONOFONICDIR" ]; then
    echo "monofonIC source not found at $MONOFONICDIR — cloning from GitHub..."
    git clone https://github.com/cosmo-sims/monofonIC.git "$MONOFONICDIR"
fi

# --- Parse command line arguments ---
BUILDDIR_DEFAULT="$REPO_ROOT/monofonic_build"
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

BUILDDIR="${BUILDDIR:-$BUILDDIR_DEFAULT}"

# --- Build ---
if [ -d "$BUILDDIR" ] && [ -f "$BUILDDIR/monofonIC" ]; then
    echo "monofonIC binary already exists in $BUILDDIR, skipping compilation"
else
    # Set up compilers / libraries
    if [[ $(uname -s) == "Darwin" ]]; then
        # macOS: Apple Clang has no OpenMP; use the newest Homebrew GNU compiler.
        BREW_BIN="$(brew --prefix 2>/dev/null)/bin"
        CC_BIN=$(ls "$BREW_BIN"/gcc-[0-9]* 2>/dev/null | grep -E 'gcc-[0-9]+$' | sort -V | tail -1)
        CXX_BIN=$(ls "$BREW_BIN"/g++-[0-9]* 2>/dev/null | grep -E 'g\+\+-[0-9]+$' | sort -V | tail -1)
        if [ -z "$CC_BIN" ] || [ -z "$CXX_BIN" ]; then
            echo "Error: no Homebrew GNU gcc/g++ found in $BREW_BIN (Apple Clang lacks OpenMP)." >&2
            echo "Install with: brew install gcc" >&2
            exit 1
        fi
        export CC="$CC_BIN"
        export CXX="$CXX_BIN"
        echo "Using compilers: CC=$CC  CXX=$CXX"
        # CMake's FindGSL / FindHDF5 do not search Homebrew prefixes by default.
        BREW_PREFIX="$(brew --prefix)"
        export GSL_ROOT_DIR="$(brew --prefix gsl)"
        # MPI build: use the parallel 'hdf5-mpi' variant (matches open-mpi + FFTW3-MPI);
        # fall back to serial 'hdf5' if that is what is installed.
        for hp in hdf5-mpi hdf5; do
            cand="$(brew --prefix "$hp" 2>/dev/null)"
            if [ -n "$cand" ] && [ -f "$cand/include/hdf5.h" ]; then export HDF5_ROOT="$cand"; break; fi
        done
        export CMAKE_PREFIX_PATH="$BREW_PREFIX:${CMAKE_PREFIX_PATH:-}"
    else
        # Cluster: load required modules. MPI is required (see the cmake flags
        # below), so this needs an MPI-enabled FFTW and a parallel HDF5. Module
        # names vary by site — adjust to match `module avail`.
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

    if [ ! -d "$BUILDDIR" ]; then
        if [ -e "$BUILDDIR" ]; then
            echo "Error: $BUILDDIR exists but is not a directory"
            exit 1
        fi
        echo "Creating build directory: $BUILDDIR"
        mkdir -p "$BUILDDIR" || { echo "Error: Failed to create $BUILDDIR"; exit 1; }
    fi

    cd "$BUILDDIR" || { echo "Error: Failed to cd to $BUILDDIR"; exit 1; }
    echo "Compiling monofonIC in $BUILDDIR..."
    cmake \
        -DENABLE_MPI=ON \
        -DENABLE_PLT=ON \
        -DENABLE_PANPHASIA=OFF \
        -DENABLE_CLASS=ON \
        "$MONOFONICDIR"
    make -j
fi
