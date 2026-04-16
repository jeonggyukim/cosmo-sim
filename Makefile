# Makefile for cosmo-pipeline C programs (sources in src/, binaries output to repo root)
# Platform detection follows the same pattern as ../tigris/prepare.sh

UNAME_S := $(shell uname -s)
UNAME_N := $(shell uname -n)
USER    := $(shell whoami)

# --- Corrfunc paths ---
# Default: ../Corrfunc (sibling of this repo). Override with:
#   make CORRFUNCDIR=/path/to/Corrfunc
# or via run_pipeline.sh --corrfunc-dir /path/to/Corrfunc
CORRFUNCDIR ?= $(realpath $(dir $(lastword $(MAKEFILE_LIST)))../Corrfunc)
CORRFUNC_INC := $(CORRFUNCDIR)/theory/xi $(CORRFUNCDIR)/utils
CORRFUNC_LIB := $(CORRFUNCDIR)/theory/xi/libcountpairs_xi.a

# --- Compiler and HDF5 paths (platform-specific) ---
ifeq ($(UNAME_S), Darwin)
    CC       := clang
    HDF5_INC := /opt/homebrew/include
    HDF5_LIB := /opt/homebrew/lib
    OMP_FLAG := -fopenmp=libomp
else ifeq ($(UNAME_N), jgkim-kias)
    CC       := gcc-13
    HDF5_INC := $(HOME)/local/include
    HDF5_LIB := $(HOME)/local/lib
    OMP_FLAG := -fopenmp
else ifeq ($(UNAME_N), jgkim-home)
    CC       := gcc-13
    HDF5_INC := $(HOME)/local/hdf5/include
    HDF5_LIB := $(HOME)/local/hdf5/lib
    OMP_FLAG := -fopenmp
else ifneq ($(filter grammar%, $(UNAME_N)),)
    CC       := gcc
    HDF5_INC := $(HOME)/libs/hdf5_gnu/include
    HDF5_LIB := $(HOME)/libs/hdf5_gnu/lib
    OMP_FLAG := -fopenmp
else
    CC       := gcc
    HDF5_INC := $(HDF5_HOME)/include
    HDF5_LIB := $(HDF5_HOME)/lib
    OMP_FLAG := -fopenmp
endif

CORRFUNC_VERSION := $(shell cd $(CORRFUNCDIR) && git describe --tags --abbrev=0 2>/dev/null || echo "2.5.3")

CFLAGS := -O3 -std=c99 -Wall $(OMP_FLAG) \
          $(addprefix -I, $(CORRFUNC_INC)) -I$(HDF5_INC) \
          -DVERSION=\"$(CORRFUNC_VERSION)\"

LDFLAGS := -L$(HDF5_LIB) -lhdf5 -lm $(OMP_FLAG)

# --- Targets ---
all: compute_xi compute_xi_cic

# Clone and build Corrfunc if the static library is missing
$(CORRFUNC_LIB):
	@echo "==> Corrfunc not found — cloning from GitHub..."
	git clone --depth=1 https://github.com/manodeep/Corrfunc.git $(CORRFUNCDIR)
	$(MAKE) -C $(CORRFUNCDIR) CC=$(CC) theory

compute_xi: src/compute_xi.c $(CORRFUNC_LIB)
	$(CC) $(CFLAGS) $< $(CORRFUNC_LIB) $(LDFLAGS) -o $@

compute_xi_cic: src/compute_xi_cic.c
	$(CC) -O3 -std=c99 -Wall $(OMP_FLAG) -I$(HDF5_INC) \
	      -L$(HDF5_LIB) -lhdf5 -lm $(OMP_FLAG) \
	      $< -o $@

clean:
	rm -f compute_xi compute_xi_cic

.PHONY: all clean
