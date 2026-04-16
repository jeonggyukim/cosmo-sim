/*
 * compute_xi.c — Compute the real-space 2-point correlation function ξ(r)
 *                for SWIFT IC particles using Corrfunc.
 *
 * Usage:
 *   ./compute_xi <ics.hdf5> <binfile> [nthreads] [-n NSUB]
 *
 *   ics.hdf5 : SWIFT HDF5 IC file containing PartType1/Coordinates and
 *              Header/BoxSize (in Mpc/h).
 *   binfile  : ASCII file with two columns (rmin rmax) per bin, one bin per
 *              line.  See rbins.txt for an example.
 *   nthreads : number of OpenMP threads (default: 4).
 *   -n NSUB  : subsample to NSUB particles before pair counting.
 *              If omitted, NSUB is chosen automatically (see below).
 *
 * Output (stdout): r_avg  r_low  r_high  xi  npairs  (one line per bin)
 * Progress/diagnostics are written to stderr.
 *
 * -----------------------------------------------------------------------
 * Estimator
 * -----------------------------------------------------------------------
 * For a periodic box with no random catalogue, Corrfunc uses the natural
 * (Peebles–Hauser) estimator:
 *
 *   ξ(r) = DD(r) / DD_rand(r) - 1
 *
 * where DD(r) is the actual data–data pair count in the shell [r, r+dr),
 * and DD_rand = N_s*(N_s-1)/2 * V_shell / V_box is the analytically
 * expected count for a spatially uniform distribution of the same N_s
 * particles.  This avoids the need to generate a random catalogue and is
 * exact in the periodic-box geometry.
 *
 * -----------------------------------------------------------------------
 * Subsampling
 * -----------------------------------------------------------------------
 * Pair counting scales as O(N_s²), so running on all N ~ 10^7 particles
 * is expensive.  We draw a random subset of N_s particles (Fisher-Yates
 * partial shuffle, fixed seed=42 for reproducibility).
 *
 * The statistical error is dominated by Poisson shot noise on DD:
 *
 *   σ(ξ) ≈ (1 + ξ) / √DD  ≈ 1/√DD   [since ξ ≈ 0 for ICs]
 *
 * Auto-sizing rule: target DD_TARGET = 10,000 pairs in the smallest
 * (innermost, hardest) bin whose shell volume is
 *
 *   V_shell_min = 4π/3 * (r_hi³ - r_lo³)
 *
 * Solving DD = N_s*(N_s-1)/2 * V_shell_min / V_box ≈ DD_TARGET gives:
 *
 *   N_s = ceil( sqrt( 2 * DD_TARGET * V_box / V_shell_min ) )
 *
 * This targets ~1% statistical precision per bin.  N_s is capped at N
 * if the full catalogue already satisfies the target.
 *
 * Reference: Peebles & Hauser (1974); Hamilton (1993).
 *
 * -----------------------------------------------------------------------
 * Build:
 *   make compute_xi
 * -----------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include <hdf5.h>

/* Corrfunc headers */
#include "countpairs_xi.h"
#include "defs.h"

#define DD_TARGET 10000.0   /* target pair count in the innermost bin for auto Nsub */

/* ------------------------------------------------------------------ */
/* HDF5 helpers                                                        */
/* ------------------------------------------------------------------ */

/* Read a scalar double attribute from an open HDF5 object (group/dataset). */
static double read_attr_double(hid_t loc, const char *name)
{
    double val;
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(attr);
    return val;
}

/*
 * read_positions — open a SWIFT IC HDF5 file and extract dark-matter
 * particle coordinates.
 *
 * The file layout expected:
 *   /Header          (group) — scalar attribute "BoxSize" in Mpc/h
 *   /PartType1/Coordinates   — float32 array of shape (N, 3), in Mpc/h
 *
 * Coordinates are stored as float32 in the file to save space; we
 * promote them to double for the Corrfunc calculation.
 *
 * Returns the particle count N and fills *X, *Y, *Z (caller must free).
 */
static int64_t read_positions(const char *fname,
                               double **X, double **Y, double **Z,
                               double *boxsize)
{
    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { fprintf(stderr, "Error: cannot open %s\n", fname); exit(EXIT_FAILURE); }

    /* Read box size from the Header group */
    hid_t hdr = H5Gopen(file, "Header", H5P_DEFAULT);
    *boxsize = read_attr_double(hdr, "BoxSize");
    H5Gclose(hdr);

    /* Open the coordinates dataset and query its shape: (N, 3) */
    hid_t dset  = H5Dopen(file, "PartType1/Coordinates", H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, NULL);
    int64_t N = (int64_t)dims[0];   /* number of particles */

    /* Read all coordinates into a flat float buffer: [x0,y0,z0, x1,y1,z1, ...] */
    float *buf = malloc(N * 3 * sizeof(float));
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Sclose(space); H5Dclose(dset); H5Fclose(file);

    /* Deinterleave into separate X/Y/Z arrays of double, as required by Corrfunc */
    *X = malloc(N * sizeof(double));
    *Y = malloc(N * sizeof(double));
    *Z = malloc(N * sizeof(double));
    for (int64_t i = 0; i < N; i++) {
        (*X)[i] = (double)buf[3*i + 0];
        (*Y)[i] = (double)buf[3*i + 1];
        (*Z)[i] = (double)buf[3*i + 2];
    }
    free(buf);
    return N;
}

/* ------------------------------------------------------------------ */
/* Read first bin edges from binfile                                   */
/* ------------------------------------------------------------------ */

/*
 * read_first_bin — parse the first line of the bin-edge file to get
 * the innermost bin [rlo, rhi).  This is the smallest shell volume and
 * therefore the bin that sets the minimum required Nsub.
 */
static void read_first_bin(const char *binfile, double *rlo, double *rhi)
{
    FILE *fp = fopen(binfile, "r");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", binfile); exit(EXIT_FAILURE); }
    if (fscanf(fp, "%lf %lf", rlo, rhi) != 2) {
        fprintf(stderr, "Error: bad binfile format\n"); exit(EXIT_FAILURE);
    }
    fclose(fp);
}

/* ------------------------------------------------------------------ */
/* Fisher-Yates partial shuffle: put first Nsub elements in random    */
/* order drawn from [0, N)                                            */
/* ------------------------------------------------------------------ */

/*
 * subsample — in-place random subsample of the coordinate arrays.
 *
 * Runs a partial Fisher-Yates shuffle with a fixed seed (42) so that
 * results are reproducible across runs.  After the call, the first Nsub
 * elements of X/Y/Z are a uniform random draw without replacement from
 * the full N-particle set.  The remaining elements are left in an
 * unspecified order and are not used by the caller.
 */
static void subsample(double *X, double *Y, double *Z,
                      int64_t N, int64_t Nsub, unsigned int seed)
{
    srand(seed);
    for (int64_t i = 0; i < Nsub; i++) {
        /* pick a random index from [i, N) and swap it into position i */
        int64_t j = i + (int64_t)rand() % (N - i);
#define SWAP(a, b) do { double _t = (a); (a) = (b); (b) = _t; } while(0)
        SWAP(X[i], X[j]);
        SWAP(Y[i], Y[j]);
        SWAP(Z[i], Z[j]);
#undef SWAP
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <ics.hdf5> <binfile> [nthreads] [-n NSUB] [-s SEED]\n", prog);
    fprintf(stderr, "  -n NSUB  subsample to NSUB particles (auto if omitted)\n");
    fprintf(stderr, "  -s SEED  random seed for subsampling (default: 42)\n");
}

int main(int argc, char **argv)
{
    if (argc < 3) { print_usage(argv[0]); return EXIT_FAILURE; }

    const char *hdf5file = argv[1];
    const char *binfile  = argv[2];
    int      nthreads = 4;    /* default thread count */
    int64_t  nsub_arg = -1;   /* -1 = auto-size Nsub */
    unsigned int seed = 42;   /* default seed for reproducibility */

    /* Parse optional arguments: positional nthreads, -n NSUB, -s SEED */
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            nsub_arg = (int64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else {
            nthreads = atoi(argv[i]);
        }
    }

    /* --- Load particle positions from the SWIFT IC file --- */
    double *X = NULL, *Y = NULL, *Z = NULL;
    double boxsize;
    int64_t N = read_positions(hdf5file, &X, &Y, &Z, &boxsize);
    double V_box = boxsize * boxsize * boxsize;   /* (Mpc/h)^3 */

    fprintf(stderr, "Loaded    : %"PRId64" particles, boxsize = %.4g Mpc/h\n", N, boxsize);

    /* --- Determine subsample size Nsub --- */
    /*
     * We only need the first bin to compute V_shell_min; the auto-sizing
     * formula targets DD_TARGET pairs in that bin (the hardest case).
     */
    double rlo, rhi;
    read_first_bin(binfile, &rlo, &rhi);
    double V_shell_min = (4.0 * M_PI / 3.0) * (rhi*rhi*rhi - rlo*rlo*rlo);

    int64_t Nsub;
    if (nsub_arg > 0) {
        /* User explicitly set -n; clamp to N */
        Nsub = (nsub_arg < N) ? nsub_arg : N;
        fprintf(stderr, "Subsample : %"PRId64" (user-specified)\n", Nsub);
    } else {
        /* Auto: solve N_s² * V_shell_min / (2 V_box) = DD_TARGET for N_s */
        Nsub = (int64_t)ceil(sqrt(2.0 * DD_TARGET * V_box / V_shell_min));
        if (Nsub > N) Nsub = N;
        fprintf(stderr, "Subsample : %"PRId64" (auto: target %.0f pairs in smallest bin)\n",
                Nsub, DD_TARGET);
    }

    /* Report predicted pair count and resulting ξ error in the innermost bin */
    double DD_pred = (double)Nsub * (double)(Nsub - 1) / 2.0 * V_shell_min / V_box;
    fprintf(stderr, "Predicted DD (smallest bin): %.0f  →  σ(ξ) ≈ %.4f\n",
            DD_pred, 1.0 / sqrt(DD_pred));

    /* --- Draw the random subsample (in-place partial shuffle) --- */
    fprintf(stderr, "Seed      : %u\n", seed);
    if (Nsub < N) subsample(X, Y, Z, N, Nsub, seed);

    fprintf(stderr, "Threads   : %d\n", nthreads);

    /* --- Run Corrfunc xi (periodic natural estimator) --- */
    /*
     * get_config_options() returns a zeroed options struct with safe defaults.
     * We enable:
     *   periodic   = 1  — use periodic boundary conditions (no randoms needed)
     *   float_type = 8  — use double precision internally
     *   verbose    = 1  — print per-bin progress to stderr
     */
    struct config_options options = get_config_options();
    options.verbose    = 1;
    options.periodic   = 1;
    options.float_type = sizeof(double);

    results_countpairs_xi results;
    int status = countpairs_xi(Nsub, X, Y, Z,
                               boxsize, nthreads, binfile,
                               &results, &options, NULL);
    free(X); free(Y); free(Z);

    if (status != EXIT_SUCCESS) {
        fprintf(stderr, "countpairs_xi failed\n");
        return EXIT_FAILURE;
    }

    /* --- Print results to stdout ---
     *
     * Corrfunc stores bin upper edges in results.rupp[0..nbin-1], where
     * rupp[0] is the inner edge of the first real bin (= the lower edge
     * of the first bin written in binfile).  We reconstruct rlow by
     * carrying rupp[i-1] forward.
     */
    printf("# r_avg   r_low   r_high   xi   npairs\n");
    double rlow = results.rupp[0];   /* lower edge of first output bin */
    for (int i = 1; i < results.nbin; i++) {
        printf("%.6e  %.6e  %.6e  %.6e  %"PRIu64"\n",
               results.ravg[i], rlow, results.rupp[i],
               results.xi[i], results.npairs[i]);
        rlow = results.rupp[i];
    }

    free_results_xi(&results);
    return EXIT_SUCCESS;
}
