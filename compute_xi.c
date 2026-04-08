/*
 * compute_xi.c — Compute the real-space 2-point correlation function ξ(r)
 *                for SWIFT IC particles using Corrfunc.
 *
 * Usage:
 *   ./compute_xi <ics.hdf5> <binfile> [nthreads] [-n NSUB]
 *
 *   binfile : ASCII file with two columns (rmin rmax) per bin, one bin per line.
 *   nthreads: number of OpenMP threads (default: 4)
 *   -n NSUB : subsample to NSUB particles before pair counting.
 *             If omitted, NSUB is chosen automatically (see below).
 *
 * Output (stdout): r_avg  r_low  r_high  xi  npairs  (one line per bin)
 *
 * -----------------------------------------------------------------------
 * Subsampling rule of thumb
 * -----------------------------------------------------------------------
 * For a periodic box the Corrfunc estimator is:
 *
 *   ξ(r) = DD(r) / DD_rand(r) - 1
 *
 * where DD_rand = N_s(N_s-1)/2 * V_shell / V_box is the expected count
 * for a uniform distribution.  The statistical error is dominated by
 * Poisson shot noise on DD:
 *
 *   σ(ξ) ≈ (1 + ξ) / √DD  ≈ 1/√DD   [since ξ≈0 for ICs]
 *
 * We target DD_target = 10,000 pairs in the smallest (hardest) bin.
 * The shell volume of the first bin is V_shell = 4π/3 (r_hi³ - r_lo³).
 * Solving DD = N_s² V_shell / (2 V_box) = DD_target gives:
 *
 *   N_s = ceil( sqrt( 2 * DD_target * V_box / V_shell_min ) )
 *
 * This gives ~1% statistical precision.  The result is capped at the
 * total particle count N if N_s > N.
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

#define DD_TARGET 10000.0   /* target pairs in the smallest bin */

/* ------------------------------------------------------------------ */
/* HDF5 helpers                                                        */
/* ------------------------------------------------------------------ */

static double read_attr_double(hid_t loc, const char *name)
{
    double val;
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(attr);
    return val;
}

static int64_t read_positions(const char *fname,
                               double **X, double **Y, double **Z,
                               double *boxsize)
{
    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { fprintf(stderr, "Error: cannot open %s\n", fname); exit(EXIT_FAILURE); }

    hid_t hdr = H5Gopen(file, "Header", H5P_DEFAULT);
    *boxsize = read_attr_double(hdr, "BoxSize");
    H5Gclose(hdr);

    hid_t dset  = H5Dopen(file, "PartType1/Coordinates", H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, NULL);
    int64_t N = (int64_t)dims[0];

    float *buf = malloc(N * 3 * sizeof(float));
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Sclose(space); H5Dclose(dset); H5Fclose(file);

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

static void subsample(double *X, double *Y, double *Z,
                      int64_t N, int64_t Nsub)
{
    srand(42);
    for (int64_t i = 0; i < Nsub; i++) {
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
    fprintf(stderr, "Usage: %s <ics.hdf5> <binfile> [nthreads] [-n NSUB]\n", prog);
    fprintf(stderr, "  -n NSUB  subsample to NSUB particles (auto if omitted)\n");
}

int main(int argc, char **argv)
{
    if (argc < 3) { print_usage(argv[0]); return EXIT_FAILURE; }

    const char *hdf5file = argv[1];
    const char *binfile  = argv[2];
    int      nthreads = 4;
    int64_t  nsub_arg = -1;   /* -1 = auto */

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            nsub_arg = (int64_t)atoll(argv[++i]);
        } else {
            nthreads = atoi(argv[i]);
        }
    }

    /* --- Load positions --- */
    double *X = NULL, *Y = NULL, *Z = NULL;
    double boxsize;
    int64_t N = read_positions(hdf5file, &X, &Y, &Z, &boxsize);
    double V_box = boxsize * boxsize * boxsize;

    fprintf(stderr, "Loaded    : %"PRId64" particles, boxsize = %.4g Mpc/h\n", N, boxsize);

    /* --- Determine subsample size --- */
    double rlo, rhi;
    read_first_bin(binfile, &rlo, &rhi);
    double V_shell_min = (4.0 * M_PI / 3.0) * (rhi*rhi*rhi - rlo*rlo*rlo);

    int64_t Nsub;
    if (nsub_arg > 0) {
        Nsub = (nsub_arg < N) ? nsub_arg : N;
        fprintf(stderr, "Subsample : %"PRId64" (user-specified)\n", Nsub);
    } else {
        /* Rule of thumb: target DD_TARGET pairs in the smallest bin */
        Nsub = (int64_t)ceil(sqrt(2.0 * DD_TARGET * V_box / V_shell_min));
        if (Nsub > N) Nsub = N;
        fprintf(stderr, "Subsample : %"PRId64" (auto: target %.0f pairs in smallest bin)\n",
                Nsub, DD_TARGET);
    }

    /* Predicted DD in smallest bin */
    double DD_pred = (double)Nsub * (double)(Nsub - 1) / 2.0 * V_shell_min / V_box;
    fprintf(stderr, "Predicted DD (smallest bin): %.0f  →  σ(ξ) ≈ %.4f\n",
            DD_pred, 1.0 / sqrt(DD_pred));

    /* --- Subsample --- */
    if (Nsub < N) subsample(X, Y, Z, N, Nsub);

    fprintf(stderr, "Threads   : %d\n", nthreads);

    /* --- Run Corrfunc xi --- */
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

    /* --- Print results --- */
    printf("# r_avg   r_low   r_high   xi   npairs\n");
    double rlow = results.rupp[0];
    for (int i = 1; i < results.nbin; i++) {
        printf("%.6e  %.6e  %.6e  %.6e  %"PRIu64"\n",
               results.ravg[i], rlow, results.rupp[i],
               results.xi[i], results.npairs[i]);
        rlow = results.rupp[i];
    }

    free_results_xi(&results);
    return EXIT_SUCCESS;
}
