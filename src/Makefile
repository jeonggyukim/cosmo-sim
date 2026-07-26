# Makefile for cosmo-sim C programs.
# Sources live alongside this Makefile in src/; binaries are written to ../bin/.
# Platform detection follows the same pattern as ../tigris/prepare.sh.

UNAME_S := $(shell uname -s)
UNAME_N := $(shell uname -n)

# Directory containing this Makefile (src/) and the repo root (one up).
SRC_DIR  := $(dir $(lastword $(MAKEFILE_LIST)))
REPO_DIR := $(SRC_DIR)..
BIN_DIR  := $(REPO_DIR)/bin

# --- Compiler and HDF5 paths (platform-specific) ---
ifeq ($(UNAME_S), Darwin)
    CC       := clang
    HDF5_INC := /opt/homebrew/include
    HDF5_LIB := /opt/homebrew/lib
    FFTW_INC := /opt/homebrew/include
    FFTW_LIB := /opt/homebrew/lib
    # Apple clang ships without OpenMP; link libomp from Homebrew.
    OMP_CFLAGS := -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include
    OMP_LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp
else ifeq ($(UNAME_N), jgkim-kias)
    CC          := gcc-13
    HDF5_INC    := $(HOME)/local/include
    HDF5_LIB    := $(HOME)/local/lib
    FFTW_INC    := $(HOME)/local/include
    FFTW_LIB    := $(HOME)/local/lib
    OMP_CFLAGS  := -fopenmp
    OMP_LDFLAGS := -fopenmp
else ifeq ($(UNAME_N), jgkim-home)
    CC          := gcc-13
    HDF5_INC    := $(HOME)/local/hdf5/include
    HDF5_LIB    := $(HOME)/local/hdf5/lib
    FFTW_INC    := $(HOME)/local/include
    FFTW_LIB    := $(HOME)/local/lib
    OMP_CFLAGS  := -fopenmp
    OMP_LDFLAGS := -fopenmp
else ifneq ($(filter grammar%, $(UNAME_N)),)
    CC          := gcc
    HDF5_INC    := $(HOME)/libs/hdf5_gnu/include
    HDF5_LIB    := $(HOME)/libs/hdf5_gnu/lib
    FFTW_INC    := $(HOME)/libs/fftw_gnu/include
    FFTW_LIB    := $(HOME)/libs/fftw_gnu/lib
    OMP_CFLAGS  := -fopenmp
    OMP_LDFLAGS := -fopenmp
else
    CC          := gcc
    HDF5_INC    := $(HDF5_HOME)/include
    HDF5_LIB    := $(HDF5_HOME)/lib
    FFTW_INC    := $(FFTW_HOME)/include
    FFTW_LIB    := $(FFTW_HOME)/lib
    OMP_CFLAGS  := -fopenmp
    OMP_LDFLAGS := -fopenmp
endif

# --- Targets ---
# compute_xi is the CIC-grid FFT-autocorrelation xi(r)/psi(r) estimator.
# It works at any redshift and needs only HDF5 and FFTW.
all: $(BIN_DIR)/compute_xi

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/compute_xi: $(SRC_DIR)compute_xi.c | $(BIN_DIR)
	$(CC) -O3 -std=c99 -Wall $(OMP_CFLAGS) -I$(HDF5_INC) -I$(FFTW_INC) \
	      -L$(HDF5_LIB) -L$(FFTW_LIB) -lhdf5 -lfftw3 -lm $(OMP_LDFLAGS) \
	      $< -o $@

clean:
	rm -f $(BIN_DIR)/compute_xi

.PHONY: all clean
