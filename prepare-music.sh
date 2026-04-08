#!/usr/bin/env bash

MUSICDIR=$HOME/Dropbox/Projects/MUSIC2
# Parse command line arguments
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

# Use default if not specified
BUILDDIR="${BUILDDIR:-$BUILDDIR_DEFAULT}"

if [ -d "$BUILDDIR" ] && [ -f "$BUILDDIR/MUSIC" ]; then
    echo "MUSIC binary already exists in $BUILDDIR, skipping compilation"
else
    if [[ $(uname -s) == "Darwin" ]]; then
        export FC=gfortran-14
    else
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

# Run MUSIC
