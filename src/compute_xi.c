/*
 * compute_xi.c — Two-point correlation function ξ(r) from a SWIFT IC HDF5
 *                    file via a CIC density grid and Landy-Szalay estimator.
 *
 * Adapted from xi_hdf5.c (KIAS next-cosmo-sim-tools).
 * MPI removed; OpenMP added for lag-loop parallelism.
 *
 * ─── Algorithm ────────────────────────────────────────────────────────────
 *
 *  1. Read all N particles from HDF5.
 *
 *  2. CIC assignment: distribute particle mass to an Ngrid³ grid using
 *     trilinear (Cloud-In-Cell) weights.  After normalising by the mean,
 *     each cell holds  den[i,j,k] = n[i,j,k] / <n>  = 1 + δ[i,j,k],
 *     where δ is the local overdensity.
 *
 *  3. Lag enumeration: build the list of all integer displacement vectors
 *     (di,dj,dk) whose Euclidean distance r = cell × sqrt(di²+dj²+dk²)
 *     falls in [rmin, rmax].  Only the "positive octant" di≥0,dj≥0,dk≥0
 *     is enumerated; the factor-of-2 symmetry is absorbed into the
 *     DD and RR accumulators.
 *
 *  4. Pair accumulation: for each lag (di,dj,dk) loop over all Ngrid³ cells
 *     and accumulate:
 *
 *       DD = Σ_{cells}  den(r) × den(r + lag)      (data–data)
 *       DR = Σ_{cells}  den(r)   [one-sided sum]    (data–random, analytic)
 *       RR = N_cell_pairs                            (random–random, analytic)
 *
 *     Multiply by 2 to account for the lag and its mirror (di,dj,dk) and
 *     (-di,-dj,-dk) having the same distance.
 *
 *  5. Landy-Szalay estimator:  ξ = (DD − 2·DR + RR) / RR
 *
 *     Expanding DD:
 *       DD/RR = <(1+δ_lo)(1+δ_hi)> = 1 + <δ_lo> + <δ_hi> + <δ_lo δ_hi>
 *             = 1 + ξ(lag)      (since <δ> = 0)
 *       DR/RR = <den_lo> = 1    (uniform random has <1+δ> = 1)
 *     So  ξ_LS = DD/RR − 2·DR/RR + 1 = (1+ξ) − 2 + 1 = ξ  ✓
 *
 *     By the Wiener-Khinchin theorem the density autocorrelation at lag r
 *     equals the inverse Fourier transform of |δ(k)|², i.e. the result is
 *     mathematically equivalent to FFT P(k) → Hankel transform → ξ(r).
 *
 * ─── Comparison with compute_xi_corrfunc.c (Corrfunc pair counting) ───────────────
 *
 *  compute_xi_corrfunc.c counts actual particle–particle pairs, subsampling to Nsub.
 *  Its shot noise is σ(ξ) ≈ 1/√DD_pairs, dominated by Poisson counting noise.
 *
 *  compute_xi.c works on the CIC density field.  The noise is
 *  σ(ξ_CIC) ≈ σ_δ² / √N_cells, where σ_δ is the rms overdensity on the grid.
 *  For a perturbed lattice at high z the particles are nearly uniformly
 *  distributed, σ_δ ≈ D(z)·σ_δ(z=0) ≪ 1, so σ(ξ_CIC) ≪ σ(ξ_Corrfunc).
 *  This is why the CIC estimator can detect ξ at high z where Corrfunc fails.
 *
 *  Schematically:
 *    | Estimator       | Uses particles | Shot noise     | Works at high z |
 *    | Corrfunc        | sampled pairs  | Poisson ~1/√DD | No              |
 *    | compute_xi  | density grid   | σ_δ²/√N_cells  | Yes             |
 *
 * ─── SWIFT IC file conventions ────────────────────────────────────────────
 *
 *  BoxSize and Coordinates are stored in Mpc (not Mpc/h).
 *  All r values in this program and in the output are therefore in Mpc.
 *  To convert to Mpc/h, multiply by H0/100.
 *
 * ─── Performance ──────────────────────────────────────────────────────────
 *
 *  Total work ≈ N_lags × Ngrid³ cell-pair multiply-adds.
 *  N_lags ≈ (4/3)π(rmax/cell)³/8  (positive octant of a sphere).
 *
 *  Example (rmax = L/2 = 512 Mpc):
 *    Ngrid=256, cell=4 Mpc  →  k_max=85, N_lags≈1.3M  → ~2×10¹³ ops (~40 min)
 *    Ngrid=128, cell=8 Mpc  →  k_max=43, N_lags≈43k   → ~9×10¹⁰ ops (~90 s)
 *
 *  OpenMP parallelises the outer lag loop (schedule dynamic, chunk 64),
 *  so wall-time scales roughly as 1/nthreads.
 *
 * ─── Build ────────────────────────────────────────────────────────────────
 *
 *   make compute_xi
 *
 * ─── Usage ────────────────────────────────────────────────────────────────
 *
 *   ./compute_xi --input data/ics_swift_n256_z2_L687.hdf5
 *   ./compute_xi --input data/ics_swift_n256_z2_L687.hdf5 \
 *            --Ngrid 128 --nbins 24 --nthreads 8 \
 *            --rmin 8.0 --rmax 512.0 \
 *            --output data/xi_cic_n256_z2_L687.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <hdf5.h>
#include <fftw3.h>

/* ─── mode enum ─────────────────────────────────────────────────────────── */
/*
 * Controls which lag directions are enumerated.
 * MODE_3D: all (di,dj,dk) — isotropic ξ(r).
 * MODE_1D_Z: only (0,0,dk) — line-of-sight ξ(s) along z.
 * MODE_1D_X/Y: similar along x/y axes.
 */
typedef enum { MODE_3D, MODE_1D_X, MODE_1D_Y, MODE_1D_Z } XiMode;

/* ─── lag vector ─────────────────────────────────────────────────────────── */
/* One element of the lag list: integer grid offsets (di,dj,dk) and the
 * corresponding Euclidean distance r = cell × sqrt(di²+dj²+dk²). */
typedef struct { int di, dj, dk; double r; } Lag;

/* ─── periodic boundary bit mask ─────────────────────────────────────────── */
/* Each bit enables periodic wrapping along that axis when computing
 * den[i+di, j+dj, k+dk].  IC boxes are periodic in all three axes (xyz=7). */
#define PERIODIC_X 1
#define PERIODIC_Y 2
#define PERIODIC_Z 4

/* ─── program options ────────────────────────────────────────────────────── */
typedef struct {
    char    input[512];
    int     ptype;        /* HDF5 PartType index; 1 = dark matter in SWIFT */
    int     Ngrid;        /* CIC grid size; -1 = auto: min(cbrt(N), 128) */
    XiMode  mode;
    int     nbins;        /* number of radial bins */
    double  rmin;         /* minimum lag in Mpc; -1 = auto: cell size */
    double  rmax;         /* maximum lag in Mpc; -1 = auto: boxsize/2 */
    char    output[512];  /* output file for xi(r) */
    int     r2_plot;      /* if set, also write xi(r)·r² as an extra column */
    int     periodic;     /* bit mask: which axes use periodic wrapping */
    int     logbin;       /* 1 = log-spaced bins (default), 0 = linear */
    int     nthreads;     /* OpenMP threads; 0 = defer to OMP_NUM_THREADS */
    int     compute_vel;  /* if set, also compute velocity correlation ψ(r) */
    char    vel_output[512]; /* output file for ψ(r); auto-derived if empty */
    int     use_fft;      /* 1 = FFT autocorrelation (default for xyz-periodic);
                             0 = direct lag-sum */
    int     order;        /* mass-assignment order: NGP=1 CIC=2 TSC=3 PCS=4 */
    int     compute_pk;   /* if set, also estimate and write P(k) */
    char    pk_output[512]; /* output file for P(k); auto-derived if empty */
    double  H0;           /* H0 [km/s/Mpc] for h = H0/100 (P(k) unit conversion) */
    int     nkbins;       /* number of log k-bins for P(k) */
} Opts;

/* ═══════════════════════════════════════════════════════════════════════════
 *  HDF5 helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * read_attr_double — read the first element of any HDF5 attribute as double.
 *
 * Handles both scalar and array attributes by reading all elements into a
 * temporary buffer and returning buf[0].  HDF5 performs type conversion
 * (e.g. float32 → float64) automatically when H5T_NATIVE_DOUBLE is used.
 */
static double read_attr_double(hid_t obj, const char *name)
{
    hid_t    attr  = H5Aopen(obj, name, H5P_DEFAULT);
    hid_t    space = H5Aget_space(attr);
    hssize_t nelem = H5Sget_simple_extent_npoints(space);
    double  *buf   = (double *)malloc((size_t)nelem * sizeof(double));
    H5Aread(attr, H5T_NATIVE_DOUBLE, buf);
    double val = buf[0];
    free(buf);
    H5Sclose(space);
    H5Aclose(attr);
    return val;
}

/*
 * read_npart — return NumPart_ThisFile[ptype] as long long.
 *
 * SWIFT ICs store this attribute as uint32 (Gadget-2 convention).  The code
 * detects the stored type and reads via H5T_NATIVE_UINT (unsigned int, 32-bit)
 * or H5T_NATIVE_LLONG (64-bit), then converts to long long.
 */
static long long read_npart(hid_t hdr, int ptype)
{
    hid_t attr   = H5Aopen(hdr, "NumPart_ThisFile", H5P_DEFAULT);
    hid_t space  = H5Aget_space(attr);
    hssize_t nelem = H5Sget_simple_extent_npoints(space);
    hid_t atype  = H5Aget_type(attr);
    hid_t native = H5Tget_native_type(atype, H5T_DIR_ASCEND);

    long long arr[6] = {0,0,0,0,0,0};
    if (H5Tequal(native, H5T_NATIVE_UINT) || H5Tequal(native, H5T_NATIVE_INT)) {
        /* SWIFT / Gadget-2 style: unsigned int array of length 6 */
        unsigned int tmp[6] = {0};
        H5Aread(attr, H5T_NATIVE_UINT, tmp);
        for (int i = 0; i < 6; i++) arr[i] = tmp[i];
    } else {
        /* Some codes write long long directly */
        H5Aread(attr, H5T_NATIVE_LLONG, arr);
    }
    H5Tclose(native); H5Tclose(atype);
    H5Sclose(space);  H5Aclose(attr);

    if (ptype < 0 || ptype >= (int)nelem) {
        fprintf(stderr, "ptype %d out of range (NumPart has %lld entries)\n",
                ptype, (long long)nelem);
        exit(1);
    }
    return arr[ptype];
}

/*
 * read_header — extract BoxSize, Redshift, and N_particles from /Header.
 *
 * BoxSize is in Mpc for SWIFT ICs (set by MUSIC2's Swift output plugin).
 * Redshift is written by MUSIC2 directly from the IC redshift parameter.
 */
static int read_header(const char *fname, int ptype,
                       double *boxsize, double *redshift, long long *N)
{
    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { fprintf(stderr, "Cannot open %s\n", fname); return -1; }

    hid_t hdr = H5Gopen2(file, "Header", H5P_DEFAULT);
    if (hdr < 0) { H5Fclose(file); return -1; }

    *boxsize  = read_attr_double(hdr, "BoxSize");
    *redshift = read_attr_double(hdr, "Redshift");
    *N        = read_npart(hdr, ptype);

    char key[32];
    snprintf(key, sizeof(key), "PartType%d", ptype);
    if (!H5Lexists(file, key, H5P_DEFAULT)) {
        fprintf(stderr, "Group %s not found in %s\n", key, fname);
        H5Gclose(hdr); H5Fclose(file); return -1;
    }
    H5Gclose(hdr);
    H5Fclose(file);
    return 0;
}

/*
 * read_coords — read all PartType{ptype}/Coordinates into a flat double array.
 *
 * The dataset has shape (N, 3); the returned buffer is laid out as
 * [x0,y0,z0, x1,y1,z1, ...] in row-major order.
 *
 * SWIFT stores coordinates as float32 to save space.  Specifying
 * H5T_NATIVE_DOUBLE as the memory type causes HDF5 to promote float32 →
 * float64 during the read, with no extra copy needed.
 */
static int read_nx3_double(const char *fname, int ptype, const char *suffix,
                           double **out, long long N, int must_exist)
{
    *out = (double *)malloc((size_t)N * 3 * sizeof(double));
    if (!*out) { fprintf(stderr, "OOM reading %s\n", suffix); return -1; }

    char dset_name[64];
    snprintf(dset_name, sizeof(dset_name), "PartType%d/%s", ptype, suffix);

    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { fprintf(stderr, "Cannot open %s\n", fname); free(*out); *out = NULL; return -1; }

    if (must_exist && !H5Lexists(file, dset_name, H5P_DEFAULT)) {
        fprintf(stderr, "Dataset %s not found in %s\n", dset_name, fname);
        H5Fclose(file); free(*out); *out = NULL; return -1;
    }

    hid_t dset = H5Dopen2(file, dset_name, H5P_DEFAULT);
    /* HDF5 auto-promotes float32 → float64 when H5T_NATIVE_DOUBLE is requested. */
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *out);
    H5Dclose(dset);
    H5Fclose(file);
    return 0;
}

static int read_coords(const char *fname, int ptype, double **coords, long long N)
{ return read_nx3_double(fname, ptype, "Coordinates", coords, N, 0); }

static int read_velocities(const char *fname, int ptype, double **vels, long long N)
{ return read_nx3_double(fname, ptype, "Velocities",  vels,   N, 1); }

/* ═══════════════════════════════════════════════════════════════════════════
 *  CIC density grid
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * cic_assign — assign particles to a density grid using Cloud-In-Cell weights.
 *
 * CIC is a mass assignment scheme (MAS) where each particle is treated as a
 * unit-mass cloud of side length equal to one cell.  The mass is distributed
 * to the 8 nearest grid nodes with trilinear weights:
 *
 *   w(dx) = 1 − |dx|/cell   for |dx| ≤ cell,  0 otherwise.
 *
 * In code: for cell-fractional position (px, py, pz), the weights to the
 * lower (tx = 1−dx) and upper (dx) neighbours along each axis are computed,
 * and the product wx·wy·wz is added to each of the 8 surrounding cells.
 *
 * Periodic boundary conditions: the double modulo ((i+di)%Ng + Ng)%Ng
 * handles particles near the box edge correctly for any integer i+di.
 *
 * After cic_assign, the grid holds raw particle counts (not overdensity).
 * The caller normalises to den = count / mean_count = 1 + δ.
 *
 * Note: the CIC MAS introduces a smoothing kernel W(k) = sinc²(k/2k_Ny)
 * per axis, suppressing power at high k.  This affects the density field
 * used here.  For P(k) estimation this would require deconvolution, but
 * for ξ(r) at large r (rmin >> cell) the suppression is negligible.
 */
/*
 * mas_axis — per-axis B-spline mass-assignment weights for one particle.
 *
 * Given a particle at grid coordinate g = position/cell, fills the `order`
 * (wrapped cell index, weight) pairs along one axis.  The interpolation order
 * selects the scheme (Sefusatti et al. 2016): NGP=1, CIC=2, TSC=3, PCS=4.  The
 * stencil spans `order` cells and the weights sum to 1.  These weight forms
 * match icpipe's _axis_stencil so the C and Python estimators agree.
 */
static inline void mas_axis(double g, int order, int Ngrid,
                            int idx[4], double w[4])
{
    if (order == 1) {                 /* NGP */
        int c = (int)floor(g + 0.5);
        idx[0] = ((c % Ngrid) + Ngrid) % Ngrid;
        w[0]   = 1.0;
    } else if (order == 2) {          /* CIC */
        int base = (int)floor(g);
        double dx = g - base;
        double ww[2] = { 1.0 - dx, dx };
        for (int m = 0; m < 2; m++) {
            int c = base + m;
            idx[m] = ((c % Ngrid) + Ngrid) % Ngrid;
            w[m]   = ww[m];
        }
    } else if (order == 3) {          /* TSC, centred on the nearest grid point */
        int n = (int)floor(g + 0.5);
        for (int m = 0; m < 3; m++) {
            int c = n + (m - 1);
            double t = fabs(g - (double)c);
            w[m]   = (t < 0.5) ? (0.75 - t*t) : (0.5 * (1.5 - t) * (1.5 - t));
            idx[m] = ((c % Ngrid) + Ngrid) % Ngrid;
        }
    } else {                          /* PCS (order 4) */
        int mm = (int)floor(g);
        for (int m = 0; m < 4; m++) {
            int c = mm + (m - 1);
            double t = fabs(g - (double)c);
            w[m]   = (t < 1.0) ? ((4.0 - 6.0*t*t + 3.0*t*t*t) / 6.0)
                               : ((2.0 - t) * (2.0 - t) * (2.0 - t) / 6.0);
            idx[m] = ((c % Ngrid) + Ngrid) % Ngrid;
        }
    }
}

/*
 * mas_assign — deposit unit mass per particle onto an Ngrid³ grid using the
 * order-p B-spline stencil.  gshift (in cell units) offsets every particle
 * before assignment: pass 0 for the base grid and 0.5 for the interlaced
 * (half-cell-shifted) grid.  After this call the grid holds raw particle
 * counts (not overdensity).
 */
static void mas_assign(const double *coords, long long npart,
                       double boxsize, int Ngrid, int order, double gshift,
                       double *grid)
{
    double cell = boxsize / Ngrid;
    long   Ng = Ngrid, Ng2 = Ng * Ng;
    for (long long p = 0; p < npart; p++) {
        int ix[4], iy[4], iz[4];
        double wx[4], wy[4], wz[4];
        mas_axis(coords[p*3+0]/cell + gshift, order, Ngrid, ix, wx);
        mas_axis(coords[p*3+1]/cell + gshift, order, Ngrid, iy, wy);
        mas_axis(coords[p*3+2]/cell + gshift, order, Ngrid, iz, wz);
        for (int a = 0; a < order; a++) {
            long baseA = (long)ix[a] * Ng2;
            for (int b = 0; b < order; b++) {
                double wxy   = wx[a] * wy[b];
                long   baseB = baseA + (long)iy[b] * Ng;
                for (int c = 0; c < order; c++)
                    grid[baseB + iz[c]] += wxy * wz[c];
            }
        }
    }
}

/*
 * mas_vel_assign — assign per-particle velocities to three grids.
 *
 * Uses mass-weighted assignment: each particle's velocity is distributed to
 * the order³ surrounding cells with the same B-spline weights as mas_assign.
 * The velocity grids therefore hold
 *
 *   vα_sum[cell] = Σ_p w_p * vα_p
 *
 * which must be divided by the mass (count) grid to obtain the mean velocity:
 *
 *   v_field[cell] = vα_sum[cell] / count[cell]
 *
 * The caller supplies the pre-built count grid (output of mas_assign before
 * normalisation) and receives three normalised velocity-field grids.  The
 * assignment order must match the one used to build count_grid.
 *
 * For cells with zero mass (empty cells — extremely rare in IC files), the
 * velocity is set to 0 (the linear-theory mean is zero, so this is unbiased).
 *
 * Note on SWIFT velocity units: MUSIC2's SWIFT plugin writes the on-disk
 * Velocities dataset in peculiar-velocity units (km/s), i.e. v_pec itself,
 * not a·v_pec.  (SWIFT's *internal* code variable is u = a·v_pec, but the
 * conversion happens at IC read time.)  So ψ(r) = ⟨v·v'⟩ comes out in
 * (km/s)² directly and is comparable to linear theory ψ_pec.
 */
static void mas_vel_assign(const double *coords, const double *vels,
                           long long npart, double boxsize, int Ngrid, int order,
                           const double *count_grid,
                           double *vx_field, double *vy_field, double *vz_field)
{
    double cell = boxsize / Ngrid;
    long   Ng = Ngrid, Ng2 = Ng * Ng;
    long   Ng3 = (long)Ngrid * Ngrid * Ngrid;

    double *vx_sum = (double *)calloc((size_t)Ng3, sizeof(double));
    double *vy_sum = (double *)calloc((size_t)Ng3, sizeof(double));
    double *vz_sum = (double *)calloc((size_t)Ng3, sizeof(double));
    if (!vx_sum || !vy_sum || !vz_sum) {
        fprintf(stderr, "OOM in mas_vel_assign\n"); exit(1);
    }

    for (long long p = 0; p < npart; p++) {
        int ix[4], iy[4], iz[4];
        double wx[4], wy[4], wz[4];
        mas_axis(coords[p*3+0]/cell, order, Ngrid, ix, wx);
        mas_axis(coords[p*3+1]/cell, order, Ngrid, iy, wy);
        mas_axis(coords[p*3+2]/cell, order, Ngrid, iz, wz);
        double vpx = vels[p*3+0], vpy = vels[p*3+1], vpz = vels[p*3+2];
        for (int a = 0; a < order; a++) {
            long baseA = (long)ix[a] * Ng2;
            for (int b = 0; b < order; b++) {
                double wxy   = wx[a] * wy[b];
                long   baseB = baseA + (long)iy[b] * Ng;
                for (int c = 0; c < order; c++) {
                    long   ii = baseB + iz[c];
                    double ww = wxy * wz[c];
                    vx_sum[ii] += ww * vpx;
                    vy_sum[ii] += ww * vpy;
                    vz_sum[ii] += ww * vpz;
                }
            }
        }
    }

    /*
     * Normalise: divide weighted sum by cell mass (particle count).
     * count_grid is the raw count (before normalisation to 1+δ), so
     * count_grid[cell] = mean_count * den[cell].  For the velocity field
     * we need count_grid[cell] itself (the denominator for the weighted mean).
     *
     * Empty cells: set velocity to 0 (linear-theory mean ≈ 0, so unbiased).
     */
    for (long i = 0; i < Ng3; i++) {
        double cnt = count_grid[i];
        vx_field[i] = (cnt > 0.0) ? vx_sum[i] / cnt : 0.0;
        vy_field[i] = (cnt > 0.0) ? vy_sum[i] / cnt : 0.0;
        vz_field[i] = (cnt > 0.0) ? vz_sum[i] / cnt : 0.0;
    }

    free(vx_sum); free(vy_sum); free(vz_sum);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Lag vector enumeration
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * build_lag_list — enumerate all integer grid displacements (di,dj,dk) whose
 *                  Euclidean distance r = sqrt(di²+dj²+dk²) × cell falls in
 *                  [rmin, rmax], for the chosen correlation mode.
 *
 * For MODE_3D we restrict to the positive octant (dk≥0, dj≥0, di≥0, with
 * the origin excluded).  Reflective symmetry means each such vector and its
 * negatives contribute identically to the sum; the factor of 2 is absorbed
 * in accumulate_pairs (DD += 2·dd_sum, RR += 2·n_pairs).
 *
 * k_max = ceil(rmax/cell) is the largest integer displacement we need to
 * consider; it is capped at Ngrid-1 to prevent wrap-around artefacts.
 *
 * For 1D modes only a single axis is non-zero, giving ξ along a grid axis
 * (useful for checking isotropy or line-of-sight statistics).
 *
 * Returns a heap-allocated Lag array of length *n_lags (caller must free).
 */
static Lag *build_lag_list(int Ngrid, double cell,
                           double rmin, double rmax,
                           XiMode mode, long *n_lags)
{
    int k_max = (int)ceil(rmax / cell);
    if (k_max >= Ngrid) k_max = Ngrid - 1;

    /* Upper bound on the number of lag vectors in the positive octant */
    long cap = (long)(k_max+1) * (k_max+1) * (k_max+1) + 10;
    Lag *lags = (Lag *)malloc((size_t)cap * sizeof(Lag));
    if (!lags) { fprintf(stderr, "OOM for lag list\n"); exit(1); }

    long n = 0;
    if (mode == MODE_3D) {
        /*
         * Iterate over the positive octant: dk, dj ≥ 0, di ≥ 0.
         * The special case (dk=0, dj=0) starts di at 1 to exclude the origin.
         */
        for (int dk = 0; dk <= k_max; dk++)
        for (int dj = 0; dj <= k_max; dj++) {
            int di_start = (dk==0 && dj==0) ? 1 : 0;
            for (int di = di_start; di <= k_max; di++) {
                double r = sqrt((double)(di*di + dj*dj + dk*dk)) * cell;
                if (r >= rmin && r <= rmax) {
                    lags[n].di=di; lags[n].dj=dj;
                    lags[n].dk=dk; lags[n].r=r; n++;
                }
            }
        }
    } else if (mode == MODE_1D_Z) {
        /* Only the (0,0,dk) direction */
        for (int dkk = 1; dkk <= k_max; dkk++) {
            double r = dkk * cell;
            if (r >= rmin && r <= rmax) {
                lags[n].di=0; lags[n].dj=0; lags[n].dk=dkk; lags[n].r=r; n++;
            }
        }
    } else if (mode == MODE_1D_Y) {
        for (int djj = 1; djj <= k_max; djj++) {
            double r = djj * cell;
            if (r >= rmin && r <= rmax) {
                lags[n].di=0; lags[n].dj=djj; lags[n].dk=0; lags[n].r=r; n++;
            }
        }
    } else { /* MODE_1D_X */
        for (int dii = 1; dii <= k_max; dii++) {
            double r = dii * cell;
            if (r >= rmin && r <= rmax) {
                lags[n].di=dii; lags[n].dj=0; lags[n].dk=0; lags[n].r=r; n++;
            }
        }
    }
    *n_lags = n;
    return lags;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Bin search
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * find_bin — binary search for r in the bin-edge array.
 *
 * bin_edges has nbins+1 elements: [e0, e1, ..., e_nbins].
 * Bin b covers [e_b, e_{b+1}).  Returns -1 if r is out of range.
 */
static inline int find_bin(const double *edges, int nbins, double r)
{
    if (r < edges[0] || r > edges[nbins]) return -1;
    int lo = 0, hi = nbins-1;
    while (lo < hi) {
        int mid = (lo+hi)/2;
        if (edges[mid+1] <= r) lo = mid+1;
        else                    hi = mid;
    }
    return (r >= edges[lo] && r <= edges[lo+1]) ? lo : -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  FFT-based autocorrelation (Wiener–Khinchin)
 *
 *  For a fully periodic box the cyclic autocorrelation
 *    C[r] = Σ_x δ(x) δ(x+r)
 *  equals IDFT(|DFT(δ)|²)[r] exactly (discrete Wiener–Khinchin).  Cost drops
 *  from O(N_lags × Ngrid³) to O(Ngrid³ log Ngrid).  See notes/cosmo_ic.tex
 *  §"Correlation function estimation: direct pair sum vs. FFT".
 *
 *  FFTW convention: r2c + c2r (unnormalised), so
 *    c2r(|r2c(δ)|²)[r] = Ngrid³ × C[r]
 *  and ⟨δ(x)δ(x+r)⟩ = C[r] / Ngrid³ = c2r_result[r] / Ngrid⁶.
 *
 *  The DD/DR/RR accumulators are filled so downstream Landy–Szalay
 *    ξ = (DD − 2·DR + RR) / RR
 *  evaluates to ⟨ξ⟩_b without modification of the main output loop.
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline int wrap_signed(int d, int N)
{
    /* Map torus index d ∈ [0, N) to the signed displacement nearest zero. */
    return (d <= N/2) ? d : d - N;
}

static void accumulate_pairs_fft(const double *den, int Ngrid, double cell,
                                 XiMode mode,
                                 const double *bin_edges, int nbins,
                                 double *DD, double *DR, double *RR)
{
    long Ng   = Ngrid;
    long Ng3  = Ng * Ng * Ng;
    long Nch  = Ng/2 + 1;            /* complex z-length after r2c */
    long Ncom = Ng * Ng * Nch;

    double       *field = (double *)fftw_malloc(sizeof(double)       * (size_t)Ng3);
    fftw_complex *fhat  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_t)Ncom);
    double       *acor  = (double *)fftw_malloc(sizeof(double)       * (size_t)Ng3);
    if (!field || !fhat || !acor) {
        fprintf(stderr, "OOM in accumulate_pairs_fft\n"); exit(1);
    }

    /* δ = den − 1  (den = 1 + δ after normalisation in main) */
    for (long i = 0; i < Ng3; i++) field[i] = den[i] - 1.0;

    fftw_plan fwd = fftw_plan_dft_r2c_3d(Ngrid, Ngrid, Ngrid,
                                         field, fhat, FFTW_ESTIMATE);
    fftw_execute(fwd);
    fftw_destroy_plan(fwd);

    /* |δ̂|² in place */
    for (long i = 0; i < Ncom; i++) {
        double re = fhat[i][0], im = fhat[i][1];
        fhat[i][0] = re*re + im*im;
        fhat[i][1] = 0.0;
    }

    fftw_plan bwd = fftw_plan_dft_c2r_3d(Ngrid, Ngrid, Ngrid,
                                         fhat, acor, FFTW_ESTIMATE);
    fftw_execute(bwd);
    fftw_destroy_plan(bwd);
    fftw_free(fhat);
    fftw_free(field);

    /* acor[lag] = Ngrid³ × Σ_x δ(x)·δ(x+lag). ⟨ξ⟩(lag) = acor[lag] / Ngrid⁶. */
    double inv_N6 = 1.0 / ((double)Ng3 * (double)Ng3);

    long   *cnt    = (long   *)calloc((size_t)nbins, sizeof(long));
    double *sum_xi = (double *)calloc((size_t)nbins, sizeof(double));
    if (!cnt || !sum_xi) { fprintf(stderr, "OOM in accumulate_pairs_fft (bins)\n"); exit(1); }

    for (int di = 0; di < Ngrid; di++) {
        int diw = wrap_signed(di, Ngrid);
        if (mode == MODE_1D_X && (diw == 0)) continue;
        if (mode == MODE_1D_Y || mode == MODE_1D_Z) if (diw != 0) continue;
        for (int dj = 0; dj < Ngrid; dj++) {
            int djw = wrap_signed(dj, Ngrid);
            if (mode == MODE_1D_Y && (djw == 0)) continue;
            if (mode == MODE_1D_X || mode == MODE_1D_Z) if (djw != 0) continue;
            long row = ((long)di*Ng + (long)dj) * Ng;
            for (int dk = 0; dk < Ngrid; dk++) {
                int dkw = wrap_signed(dk, Ngrid);
                if (mode == MODE_1D_Z && (dkw == 0)) continue;
                if (mode == MODE_1D_X || mode == MODE_1D_Y) if (dkw != 0) continue;
                if (di == 0 && dj == 0 && dk == 0) continue;
                double r = sqrt((double)(diw*diw + djw*djw + dkw*dkw)) * cell;
                int b = find_bin(bin_edges, nbins, r);
                if (b < 0) continue;
                double xi_lag = acor[row + dk] * inv_N6;
                sum_xi[b] += xi_lag;
                cnt[b]++;
            }
        }
    }
    fftw_free(acor);

    /*
     * Fill DD/DR/RR such that (DD − 2·DR + RR)/RR == sum_xi/cnt = ⟨ξ⟩_b:
     *   RR[b] = cnt[b] × Ngrid³
     *   DR[b] = cnt[b] × Ngrid³
     *   DD[b] = Ngrid³ × (cnt[b] + sum_xi[b])
     * Derivation: per lag, Σ_x den·den' = Ngrid³·(1+ξ), Σ_x den = Ngrid³,
     *             Σ_x 1 = Ngrid³; accumulated over full-torus lags in bin b.
     */
    for (int b = 0; b < nbins; b++) {
        double rr = (double)cnt[b] * (double)Ng3;
        RR[b] = rr;
        DR[b] = rr;
        DD[b] = (double)Ng3 * ((double)cnt[b] + sum_xi[b]);
    }
    free(cnt); free(sum_xi);
}

static void accumulate_vel_pairs_fft(const double *vx, const double *vy,
                                     const double *vz, int Ngrid, double cell,
                                     XiMode mode,
                                     const double *bin_edges, int nbins,
                                     double *VV, double *RR_vel)
{
    long Ng   = Ngrid;
    long Ng3  = Ng * Ng * Ng;
    long Nch  = Ng/2 + 1;
    long Ncom = Ng * Ng * Nch;

    double       *field   = (double *)fftw_malloc(sizeof(double)       * (size_t)Ng3);
    fftw_complex *fhat    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_t)Ncom);
    fftw_complex *pow_sum = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_t)Ncom);
    double       *acor    = (double *)fftw_malloc(sizeof(double)       * (size_t)Ng3);
    if (!field || !fhat || !pow_sum || !acor) {
        fprintf(stderr, "OOM in accumulate_vel_pairs_fft\n"); exit(1);
    }

    for (long i = 0; i < Ncom; i++) { pow_sum[i][0] = 0.0; pow_sum[i][1] = 0.0; }

    fftw_plan fwd = fftw_plan_dft_r2c_3d(Ngrid, Ngrid, Ngrid,
                                         field, fhat, FFTW_ESTIMATE);

    const double *vptr[3] = {vx, vy, vz};
    for (int a = 0; a < 3; a++) {
        for (long i = 0; i < Ng3; i++) field[i] = vptr[a][i];
        fftw_execute_dft_r2c(fwd, field, fhat);
        for (long i = 0; i < Ncom; i++) {
            double re = fhat[i][0], im = fhat[i][1];
            pow_sum[i][0] += re*re + im*im;
        }
    }
    fftw_destroy_plan(fwd);
    fftw_free(fhat);
    fftw_free(field);

    fftw_plan bwd = fftw_plan_dft_c2r_3d(Ngrid, Ngrid, Ngrid,
                                         pow_sum, acor, FFTW_ESTIMATE);
    fftw_execute(bwd);
    fftw_destroy_plan(bwd);
    fftw_free(pow_sum);

    /* acor[lag] = Ngrid³ × Σ_x Σ_α v_α(x) v_α(x+lag). */
    double inv_N = 1.0 / (double)Ng3;

    long   *cnt    = (long   *)calloc((size_t)nbins, sizeof(long));
    double *sum_vv = (double *)calloc((size_t)nbins, sizeof(double));
    if (!cnt || !sum_vv) { fprintf(stderr, "OOM in accumulate_vel_pairs_fft (bins)\n"); exit(1); }

    for (int di = 0; di < Ngrid; di++) {
        int diw = wrap_signed(di, Ngrid);
        if (mode == MODE_1D_X && (diw == 0)) continue;
        if (mode == MODE_1D_Y || mode == MODE_1D_Z) if (diw != 0) continue;
        for (int dj = 0; dj < Ngrid; dj++) {
            int djw = wrap_signed(dj, Ngrid);
            if (mode == MODE_1D_Y && (djw == 0)) continue;
            if (mode == MODE_1D_X || mode == MODE_1D_Z) if (djw != 0) continue;
            long row = ((long)di*Ng + (long)dj) * Ng;
            for (int dk = 0; dk < Ngrid; dk++) {
                int dkw = wrap_signed(dk, Ngrid);
                if (mode == MODE_1D_Z && (dkw == 0)) continue;
                if (mode == MODE_1D_X || mode == MODE_1D_Y) if (dkw != 0) continue;
                if (di == 0 && dj == 0 && dk == 0) continue;
                double r = sqrt((double)(diw*diw + djw*djw + dkw*dkw)) * cell;
                int b = find_bin(bin_edges, nbins, r);
                if (b < 0) continue;
                sum_vv[b] += acor[row + dk] * inv_N;  /* = Σ_x v(x)·v(x+lag) */
                cnt[b]++;
            }
        }
    }
    fftw_free(acor);

    /* VV/RR_vel fills so that ψ = VV/RR_vel = sum_vv / (cnt × Ngrid³) = ⟨v·v'⟩_b. */
    for (int b = 0; b < nbins; b++) {
        VV[b]     = sum_vv[b];
        RR_vel[b] = (double)cnt[b] * (double)Ng3;
    }
    free(cnt); free(sum_vv);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Pair accumulation — OpenMP parallel over lag vectors
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * accumulate_pairs — compute DD, DR, RR for all lags and bin them.
 *
 * For each lag (di,dj,dk) in the lag list, we scan all Ngrid³ cells and
 * accumulate:
 *
 *   dd_sum = Σ_cells  den[i,j,k] × den[i+di, j+dj, k+dk]
 *   lo_sum = Σ_cells  den[i,j,k]       (data, "low" side)
 *   hi_sum = Σ_cells  den[i+di,j+dj,k+dk]   (data, "high" side)
 *   n_pairs = number of valid (cell, shifted-cell) pairs
 *
 * Contributions to the bin covering that lag's r:
 *   DD[b] += 2 × dd_sum     ← factor 2: lag and its mirror contribute equally
 *   DR[b] += lo_sum + hi_sum ← both halves of the random sum
 *   RR[b] += 2 × n_pairs    ← same factor 2
 *
 * For a fully periodic box: nx = ny = nz = Ngrid, n_pairs = Ngrid³,
 * and the index arithmetic wraps modulo Ngrid.  For non-periodic axes
 * nx = Ngrid − di, etc., reducing the sum to avoid out-of-bounds access.
 *
 * OpenMP parallelism:
 *   Each thread accumulates into a private copy of DD/DR/RR (one row per
 *   thread in the flat arrays DD_t, DR_t, RR_t).  This avoids false sharing
 *   and race conditions on the bin arrays without needing atomic operations.
 *   Dynamic scheduling (chunk 64) balances load because lags at large r have
 *   more distinct distances and shorter inner loops (smaller shells), while
 *   lags near rmin have longer inner loops but fewer unique distances.
 *   After the parallel loop the per-thread arrays are summed serially.
 */
static void accumulate_pairs(const double *den, int Ngrid,
                             const Lag *lags, long n_lags,
                             const double *bin_edges, int nbins,
                             double *DD, double *DR, double *RR,
                             int periodic)
{
    long Ng  = Ngrid;
    long Ng2 = Ng * Ng;

    int per_x = (periodic & PERIODIC_X) != 0;
    int per_y = (periodic & PERIODIC_Y) != 0;
    int per_z = (periodic & PERIODIC_Z) != 0;

    /* Allocate per-thread accumulator arrays (nthreads × nbins) */
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    double *DD_t = (double *)calloc((size_t)nthreads * nbins, sizeof(double));
    double *DR_t = (double *)calloc((size_t)nthreads * nbins, sizeof(double));
    double *RR_t = (double *)calloc((size_t)nthreads * nbins, sizeof(double));
    if (!DD_t || !DR_t || !RR_t) { fprintf(stderr, "OOM in accumulate_pairs\n"); exit(1); }

#pragma omp parallel for schedule(dynamic, 64)
    for (long l = 0; l < n_lags; l++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        /* Each thread writes to its own private row */
        double *my_DD = DD_t + (size_t)tid * nbins;
        double *my_DR = DR_t + (size_t)tid * nbins;
        double *my_RR = RR_t + (size_t)tid * nbins;

        int di = lags[l].di, dj = lags[l].dj, dk = lags[l].dk;

        /* Locate the r bin for this lag vector */
        int b = find_bin(bin_edges, nbins, lags[l].r);
        if (b < 0) continue;   /* r outside [rmin,rmax] — skip */

        /*
         * Loop extent per axis:
         *   periodic → all Ng cells (high-side wraps with %)
         *   non-periodic → only Ng-d cells (high-side = low+d, stays in bounds)
         */
        long nx = per_x ? Ng : (Ng - di);
        long ny = per_y ? Ng : (Ng - dj);
        long nz = per_z ? Ng : (Ng - dk);

        double dd_sum = 0.0, lo_sum = 0.0, hi_sum = 0.0;

        for (long i = 0; i < nx; i++) {
            long ihi = per_x ? (i+di)%Ng : (i+di);
            for (long j = 0; j < ny; j++) {
                long jhi = per_y ? (j+dj)%Ng : (j+dj);
                /*
                 * Precompute row pointers to avoid repeated index arithmetic
                 * in the innermost (k) loop — the compiler can then keep
                 * lo_row and hi_row in registers or cache.
                 */
                const double *lo_row = den + i  *Ng2 + j  *Ng;
                const double *hi_row = den + ihi*Ng2 + jhi*Ng;
                /*
                 * Split into a no-wrap segment [0, nz_flat) + a wrap
                 * segment [nz_flat, nz).  The no-wrap segment has hi = lo+dk
                 * (no branch, unit stride) and vectorises under -O3.
                 */
                long nz_flat = per_z ? (Ng - dk) : nz;
                if (nz_flat > nz) nz_flat = nz;
                #pragma omp simd reduction(+:dd_sum,lo_sum,hi_sum)
                for (long k = 0; k < nz_flat; k++) {
                    double lo_v = lo_row[k];
                    double hi_v = hi_row[k + dk];
                    dd_sum += lo_v * hi_v;
                    lo_sum += lo_v;
                    hi_sum += hi_v;
                }
                for (long k = nz_flat; k < nz; k++) {
                    long khi = (k + dk) - Ng;     /* per_z implied here */
                    double lo_v = lo_row[k];
                    double hi_v = hi_row[khi];
                    dd_sum += lo_v * hi_v;
                    lo_sum += lo_v;
                    hi_sum += hi_v;
                }
            }
        }

        /* Factor 2: the mirror lag (-di,-dj,-dk) gives the same sum */
        my_DD[b] += 2.0 * dd_sum;
        my_DR[b] += lo_sum + hi_sum;       /* both halves contribute to DR */
        my_RR[b] += 2.0 * (double)(nx * ny * nz);
    }

    /* Serial reduction: sum per-thread rows into the output arrays */
    for (int t = 0; t < nthreads; t++) {
        for (int b = 0; b < nbins; b++) {
            DD[b] += DD_t[(size_t)t * nbins + b];
            DR[b] += DR_t[(size_t)t * nbins + b];
            RR[b] += RR_t[(size_t)t * nbins + b];
        }
    }
    free(DD_t); free(DR_t); free(RR_t);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Velocity pair accumulation — ψ(r) = ⟨v(x)·v(x+r)⟩
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * accumulate_vel_pairs — compute the isotropic velocity autocorrelation ψ(r).
 *
 * For each lag (di,dj,dk), accumulates the scalar product of velocity vectors
 * at the two ends of the lag across all grid cells:
 *
 *   vv_sum(lag) = Σ_cells  [vx(lo)·vx(hi) + vy(lo)·vy(hi) + vz(lo)·vz(hi)]
 *
 * Then ψ(r) = vv_sum / N_cell_pairs  (normalised per cell pair).
 *
 * This is the estimator of the 3D isotropic velocity autocorrelation:
 *
 *   ψ(r) ≡ ⟨v(x)·v(x+r)⟩ = ψ_x(r) + ψ_y(r) + ψ_z(r)
 *
 * In linear theory (for an irrotational velocity field):
 *
 *   ψ(r) = [a·H(z)·f(z)]² / (2π²) ∫₀^∞ dk P(k,z) j₀(kr)
 *
 * where f = d ln D / d ln a ≈ Ω_m(z)^0.55 and P(k,z) is the matter power
 * spectrum.  This is the same Hankel transform as ξ(r) but without the k²
 * factor — so ψ(r) is dominated by large-scale modes and is more sensitive
 * to the BAO bump than ξ(r) from density.
 *
 * No Landy-Szalay correction is needed because ⟨v⟩ = 0 in linear theory;
 * the estimator simplifies to a plain normalised cross-correlation.
 *
 * Units: ψ(r) is in (km/s)² since SWIFT velocities are stored in km/s.
 *
 * OpenMP strategy: same per-thread accumulator pattern as accumulate_pairs.
 * Each thread accumulates into a private VV row; serial reduction at the end.
 */
static void accumulate_vel_pairs(const double *vx, const double *vy, const double *vz,
                                 int Ngrid,
                                 const Lag *lags, long n_lags,
                                 const double *bin_edges, int nbins,
                                 double *VV, double *RR_vel,
                                 int periodic)
{
    long Ng  = Ngrid;
    long Ng2 = Ng * Ng;

    int per_x = (periodic & PERIODIC_X) != 0;
    int per_y = (periodic & PERIODIC_Y) != 0;
    int per_z = (periodic & PERIODIC_Z) != 0;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    /* Per-thread accumulators for VV (sum of v·v') and RR (cell-pair count) */
    double *VV_t  = (double *)calloc((size_t)nthreads * nbins, sizeof(double));
    double *RR_t  = (double *)calloc((size_t)nthreads * nbins, sizeof(double));
    if (!VV_t || !RR_t) { fprintf(stderr, "OOM in accumulate_vel_pairs\n"); exit(1); }

#pragma omp parallel for schedule(dynamic, 64)
    for (long l = 0; l < n_lags; l++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *my_VV = VV_t + (size_t)tid * nbins;
        double *my_RR = RR_t + (size_t)tid * nbins;

        int di = lags[l].di, dj = lags[l].dj, dk = lags[l].dk;

        int b = find_bin(bin_edges, nbins, lags[l].r);
        if (b < 0) continue;

        long nx = per_x ? Ng : (Ng - di);
        long ny = per_y ? Ng : (Ng - dj);
        long nz = per_z ? Ng : (Ng - dk);

        double vv_sum = 0.0;

        for (long i = 0; i < nx; i++) {
            long ihi = per_x ? (i+di)%Ng : (i+di);
            for (long j = 0; j < ny; j++) {
                long jhi = per_y ? (j+dj)%Ng : (j+dj);
                /*
                 * Row-pointer optimisation: precompute base offsets for the
                 * (i,j) and (ihi,jhi) rows so the inner k-loop is a simple
                 * stride-1 access.
                 */
                const double *vx_lo = vx + i  *Ng2 + j  *Ng;
                const double *vy_lo = vy + i  *Ng2 + j  *Ng;
                const double *vz_lo = vz + i  *Ng2 + j  *Ng;
                const double *vx_hi = vx + ihi*Ng2 + jhi*Ng;
                const double *vy_hi = vy + ihi*Ng2 + jhi*Ng;
                const double *vz_hi = vz + ihi*Ng2 + jhi*Ng;
                long nz_flat = per_z ? (Ng - dk) : nz;
                if (nz_flat > nz) nz_flat = nz;
                #pragma omp simd reduction(+:vv_sum)
                for (long k = 0; k < nz_flat; k++) {
                    vv_sum += vx_lo[k]*vx_hi[k+dk]
                            + vy_lo[k]*vy_hi[k+dk]
                            + vz_lo[k]*vz_hi[k+dk];
                }
                for (long k = nz_flat; k < nz; k++) {
                    long khi = (k + dk) - Ng;
                    vv_sum += vx_lo[k]*vx_hi[khi]
                            + vy_lo[k]*vy_hi[khi]
                            + vz_lo[k]*vz_hi[khi];
                }
            }
        }

        /* Factor 2: mirror lag (-di,-dj,-dk) gives the same vv_sum */
        my_VV[b] += 2.0 * vv_sum;
        my_RR[b] += 2.0 * (double)(nx * ny * nz);
    }

    /* Reduce per-thread rows */
    for (int t = 0; t < nthreads; t++) {
        for (int b = 0; b < nbins; b++) {
            VV[b]     += VV_t[(size_t)t * nbins + b];
            RR_vel[b] += RR_t[(size_t)t * nbins + b];
        }
    }
    free(VV_t); free(RR_t);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  P(k) estimation  (consistent with icpipe.ICField.power("delta"))
 *
 *  Interlaced order-p mass assignment + squared-window deconvolution, matching
 *  compute_pk.py so the C and Python density-P(k) estimators agree.  The same
 *  |δ(k)|² that the ξ(r) autocorrelation forms internally is here spherically
 *  binned in |k|, deconvolved, and written as a pk_*.txt table (same columns
 *  and units as compute_pk.py).
 *
 *    P(k)   = ⟨ |δ_k|² ⟩_shell · V · h³ / N² / W²(k)        [(Mpc/h)³]
 *    δ_k    = ½( F₁ + e^{iπ(nx+ny+nz)/Ng} F₂ ) − N·δ_{k,0}   (interlaced)
 *    W²(k)  = ∏_a sinc(n_a/Ng)^{2·order}     (numpy sinc = sin(πx)/(πx))
 *    P_shot = V / N · h³
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char *order_name(int order)
{
    switch (order) {
        case 1:  return "ngp";
        case 2:  return "cic";
        case 3:  return "tsc";
        default: return "pcs";
    }
}

static inline double sinc_norm(double x)   /* numpy sinc: sin(πx)/(πx), =1 at 0 */
{
    if (x == 0.0) return 1.0;
    double px = M_PI * x;
    return sin(px) / px;
}

static void compute_and_write_pk(const double *coords, long long N_total,
                                 double boxsize, int Ngrid, int order,
                                 double h, int nkbins, double redshift,
                                 int ptype, const char *input,
                                 const char *assign_name, const char *pk_path)
{
    long Ng   = Ngrid;
    long Ng3  = Ng * Ng * Ng;
    long Nch  = Ng/2 + 1;
    long Ncom = Ng * Ng * Nch;

    double V      = boxsize * boxsize * boxsize;        /* Mpc³ */
    double h3     = h * h * h;
    double kf     = 2.0 * M_PI / boxsize;               /* Mpc⁻¹ */
    double knyq_mpch  = (M_PI * Ngrid / boxsize) / h;   /* h/Mpc */
    double kfund_mpch = 2.0 * M_PI / (boxsize * h);     /* h/Mpc */
    double kmin   = kfund_mpch;
    double kmax   = 0.9 * knyq_mpch;
    double P_shot = (V / (double)N_total) * h3;         /* (Mpc/h)³ */

    /* Two half-cell-shifted count grids for interlacing (Sefusatti+ 2016). */
    double       *rho1 = (double *)fftw_malloc(sizeof(double)       * (size_t)Ng3);
    double       *rho2 = (double *)fftw_malloc(sizeof(double)       * (size_t)Ng3);
    fftw_complex *F1   = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_t)Ncom);
    fftw_complex *F2   = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_t)Ncom);
    if (!rho1 || !rho2 || !F1 || !F2) {
        fprintf(stderr, "OOM in compute_and_write_pk\n"); exit(1);
    }
    for (long i = 0; i < Ng3; i++) { rho1[i] = 0.0; rho2[i] = 0.0; }

    mas_assign(coords, N_total, boxsize, Ngrid, order, 0.0, rho1);
    mas_assign(coords, N_total, boxsize, Ngrid, order, 0.5, rho2);

    fftw_plan p1 = fftw_plan_dft_r2c_3d(Ngrid, Ngrid, Ngrid, rho1, F1, FFTW_ESTIMATE);
    fftw_execute(p1); fftw_destroy_plan(p1);
    fftw_plan p2 = fftw_plan_dft_r2c_3d(Ngrid, Ngrid, Ngrid, rho2, F2, FFTW_ESTIMATE);
    fftw_execute(p2); fftw_destroy_plan(p2);
    fftw_free(rho1); fftw_free(rho2);

    /* Log-spaced |k| bin edges in h/Mpc (matches log_bin_edges). */
    double *edges = (double *)malloc((size_t)(nkbins+1) * sizeof(double));
    double llo = log10(kmin), lhi = log10(kmax);
    for (int b = 0; b <= nkbins; b++)
        edges[b] = pow(10.0, llo + (lhi - llo) * b / nkbins);

    double *sum_k   = (double *)calloc((size_t)nkbins, sizeof(double));
    double *sum_P   = (double *)calloc((size_t)nkbins, sizeof(double));
    double *sum_P2  = (double *)calloc((size_t)nkbins, sizeof(double));
    double *sum_Pnd = (double *)calloc((size_t)nkbins, sizeof(double));
    long   *cnt     = (long   *)calloc((size_t)nkbins, sizeof(long));
    if (!edges || !sum_k || !sum_P || !sum_P2 || !sum_Pnd || !cnt) {
        fprintf(stderr, "OOM in compute_and_write_pk (bins)\n"); exit(1);
    }

    double invN2 = 1.0 / ((double)N_total * (double)N_total);

    for (long i = 0; i < Ng; i++) {
        int nx = (i < Ng/2) ? (int)i : (int)(i - Ng);
        for (long j = 0; j < Ng; j++) {
            int ny = (j < Ng/2) ? (int)j : (int)(j - Ng);
            for (long kk = 0; kk < Nch; kk++) {
                int  nz  = (int)kk;
                long idx = (i * Ng + j) * Nch + kk;

                /* Interlace combine: F = ½(F₁ + e^{iπ(nx+ny+nz)/Ng} F₂). */
                double ang = M_PI * (double)(nx + ny + nz) / (double)Ng;
                double cph = cos(ang), sph = sin(ang);
                double f2r = F2[idx][0], f2i = F2[idx][1];
                double re = 0.5 * (F1[idx][0] + (cph*f2r - sph*f2i));
                double im = 0.5 * (F1[idx][1] + (sph*f2r + cph*f2i));
                if (i == 0 && j == 0 && kk == 0) re -= (double)N_total;  /* DC subtract */

                double mag2 = re*re + im*im;
                double Pnd  = mag2 * V * h3 * invN2;      /* no deconvolution */

                double W  = sinc_norm((double)nx/(double)Ng)
                          * sinc_norm((double)ny/(double)Ng)
                          * sinc_norm((double)nz/(double)Ng);
                double W2 = 1.0;
                for (int e = 0; e < 2*order; e++) W2 *= W;
                if (i == 0 && j == 0 && kk == 0) W2 = 1.0;
                double P = Pnd / W2;

                double kmag = kf * sqrt((double)nx*nx + (double)ny*ny + (double)nz*nz) / h;
                int    b    = find_bin(edges, nkbins, kmag);
                if (b < 0) continue;      /* excludes k=0 and modes outside [kmin,kmax) */
                sum_k[b]   += kmag;
                sum_P[b]   += P;
                sum_P2[b]  += P * P;
                sum_Pnd[b] += Pnd;
                cnt[b]++;
            }
        }
    }
    fftw_free(F1); fftw_free(F2);

    FILE *fp = fopen(pk_path, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open %s for writing\n", pk_path);
        free(edges); free(sum_k); free(sum_P); free(sum_P2); free(sum_Pnd); free(cnt);
        return;
    }
    fprintf(fp,
        "# P(k) from %s   |  compute_xi.c --pk-output\n"
        "# Method : interlaced order-%d mass assignment + W^2 deconvolution\n"
        "#          (consistent with icpipe.ICField.power / compute_pk.py)\n"
        "# BoxSize=%.4g Mpc/h  N=%lld  ngrid=%d\n"
        "# Assignment: %s (order %d)   PartType: %d   z: %.6f\n"
        "# P_shot(m=1) = %.4e (Mpc/h)^3  (NOT subtracted; shown separately)\n"
        "# Columns: k[h/Mpc]  P_raw[(Mpc/h)^3]  P_shot_sub[(Mpc/h)^3]  "
        "P_nodeconv[(Mpc/h)^3]  sigma_P  nmodes  fold_m\n",
        input, order, boxsize * h, N_total, Ngrid,
        assign_name, order, ptype, redshift, P_shot);

    for (int b = 0; b < nkbins; b++) {
        if (cnt[b] == 0) continue;                 /* drop empty bins */
        double kc    = sum_k[b] / cnt[b];
        double Pmean = sum_P[b] / cnt[b];
        double Pnd   = sum_Pnd[b] / cnt[b];
        double popvar = sum_P2[b] / cnt[b] - Pmean * Pmean;   /* ddof=0, as numpy .std() */
        if (popvar < 0.0) popvar = 0.0;
        double sigma = (cnt[b] > 1) ? sqrt(popvar / (double)cnt[b]) : 0.0;
        double Pss   = Pmean - P_shot;
        fprintf(fp, "%.6e %.6e %.6e %.6e %.6e %ld %d\n",
                kc, Pmean, Pss, Pnd, sigma, cnt[b], 1);
    }
    fclose(fp);
    printf("Saved → %s  (%d k-bins)\n", pk_path, nkbins);

    free(edges); free(sum_k); free(sum_P); free(sum_P2); free(sum_Pnd); free(cnt);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Argument parsing
 * ═══════════════════════════════════════════════════════════════════════════ */

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --input FILE [options]\n"
        "\n"
        "  --input FILE       SWIFT IC HDF5 file (required)\n"
        "  --ptype INT        PartType index (default 1 = dark matter)\n"
        "  --Ngrid INT        CIC grid size (default: cbrt(N))\n"
        "                     Work ~ Ngrid³ log Ngrid with --fft (default).\n"
        "  --mode  STR        3d | 1d-x | 1d-y | 1d-z (default 3d)\n"
        "  --nbins INT        number of r bins (default 60)\n"
        "  --rmin  FLOAT      minimum lag in Mpc (default: cell size)\n"
        "  --rmax  FLOAT      maximum lag in Mpc (default: boxsize/2)\n"
        "  --output FILE      output text file for xi(r) (default xi_cic_out.txt)\n"
        "  --r2               also write xi(r)·r² as an extra column\n"
        "  --logbin           log-spaced bins (default)\n"
        "  --linear           linear-spaced bins\n"
        "  --nthreads INT     OpenMP threads (default: OMP_NUM_THREADS)\n"
        "  --periodic STR     none|x|y|z|xy|xz|yz|xyz (default: xyz)\n"
        "  --vel              also compute velocity correlation ψ(r) = <v·v'>\n"
        "  --vel-output FILE  output file for ψ(r) (default: auto-derived from --output)\n"
        "  --no-fft           force direct O(N_lags × Ngrid³) lag sum instead of\n"
        "                     O(Ngrid³ log Ngrid) FFT (FFT is default when periodic=xyz)\n"
        "  --assignment STR   mass assignment: ngp|cic|tsc|pcs (default pcs;\n"
        "                     order 1/2/3/4 B-spline, Sefusatti+ 2016)\n"
        "  --pk               also estimate P(k); auto-names pk_<stem>.txt from --output\n"
        "  --pk-output FILE   also estimate P(k), writing to FILE\n"
        "  --nkbins INT       number of log k-bins for P(k) (default 60)\n"
        "  --H0 FLOAT         H0 [km/s/Mpc] for h=H0/100 in P(k) units (default 67.11)\n"
        "  --help\n"
        "\n"
        "  r and BoxSize are in Mpc (SWIFT IC units, not Mpc/h).\n"
        "  xi output columns:  r_avg  r_low  r_high  xi(r)  DD  DR  RR\n"
        "  psi output columns: r_avg  r_low  r_high  psi(r) [(km/s)^2]\n",
        prog);
    exit(1);
}

static Opts parse_args(int argc, char **argv)
{
    Opts o;
    memset(&o, 0, sizeof(o));
    o.ptype    = 1;
    o.Ngrid    = -1;       /* -1 triggers auto-detection in main */
    o.mode     = MODE_3D;
    o.nbins    = 60;
    o.rmin     = -1.0;     /* -1 → set to cell size after header is read */
    o.rmax     = -1.0;     /* -1 → set to boxsize/3 after header is read */
    o.periodic    = PERIODIC_X | PERIODIC_Y | PERIODIC_Z;  /* xyz = all periodic */
    o.logbin      = 1;        /* default: log-spaced bins */
    o.nthreads    = 0;        /* 0 → use OMP_NUM_THREADS environment variable */
    o.compute_vel = 0;        /* default: density xi only */
    o.use_fft     = 1;        /* default: FFT path (falls back to direct if non-periodic) */
    o.order       = 4;        /* default: PCS (matches compute_pk.py) */
    o.compute_pk  = 0;        /* default: no P(k) output */
    o.H0          = 67.11;    /* default h = 0.6711 (CV_22 cosmology) */
    o.nkbins      = 60;       /* default: 60 log k-bins (matches compute_pk.py) */
    strcpy(o.output,     "xi_cic_out.txt");
    o.vel_output[0] = '\0';   /* empty: auto-derived from --output in main() */
    o.pk_output[0]  = '\0';   /* empty: auto-derived from --output in main() */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help"))                      usage(argv[0]);
        else if (!strcmp(argv[i], "--input")    && i+1<argc) strncpy(o.input,  argv[++i], 511);
        else if (!strcmp(argv[i], "--output")   && i+1<argc) strncpy(o.output, argv[++i], 511);
        else if (!strcmp(argv[i], "--ptype")    && i+1<argc) o.ptype    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--Ngrid")    && i+1<argc) o.Ngrid    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nbins")    && i+1<argc) o.nbins    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rmin")     && i+1<argc) o.rmin     = atof(argv[++i]);
        else if (!strcmp(argv[i], "--rmax")     && i+1<argc) o.rmax     = atof(argv[++i]);
        else if (!strcmp(argv[i], "--nthreads") && i+1<argc) o.nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--r2"))                    o.r2_plot     = 1;
        else if (!strcmp(argv[i], "--logbin"))                o.logbin      = 1;
        else if (!strcmp(argv[i], "--linear"))                o.logbin      = 0;
        else if (!strcmp(argv[i], "--vel"))                   o.compute_vel = 1;
        else if (!strcmp(argv[i], "--vel-output") && i+1<argc) strncpy(o.vel_output, argv[++i], 511);
        else if (!strcmp(argv[i], "--no-fft"))                o.use_fft     = 0;
        else if (!strcmp(argv[i], "--fft"))                   o.use_fft     = 1;
        else if (!strcmp(argv[i], "--pk"))                    o.compute_pk  = 1;
        else if (!strcmp(argv[i], "--pk-output") && i+1<argc) { strncpy(o.pk_output, argv[++i], 511); o.compute_pk = 1; }
        else if (!strcmp(argv[i], "--H0")       && i+1<argc)  o.H0     = atof(argv[++i]);
        else if (!strcmp(argv[i], "--nkbins")   && i+1<argc)  o.nkbins = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--assignment") && i+1<argc) {
            i++;
            if      (!strcmp(argv[i],"ngp")) o.order = 1;
            else if (!strcmp(argv[i],"cic")) o.order = 2;
            else if (!strcmp(argv[i],"tsc")) o.order = 3;
            else if (!strcmp(argv[i],"pcs")) o.order = 4;
            else { fprintf(stderr,"Unknown assignment: %s (use ngp|cic|tsc|pcs)\n", argv[i]); exit(1); }
        }
        else if (!strcmp(argv[i], "--periodic") && i+1<argc) {
            i++;
            o.periodic = 0;
            if (strchr(argv[i],'x') || strchr(argv[i],'X')) o.periodic |= PERIODIC_X;
            if (strchr(argv[i],'y') || strchr(argv[i],'Y')) o.periodic |= PERIODIC_Y;
            if (strchr(argv[i],'z') || strchr(argv[i],'Z')) o.periodic |= PERIODIC_Z;
            /* "none" or unrecognised string → 0 (already set) */
        }
        else if (!strcmp(argv[i], "--mode") && i+1<argc) {
            i++;
            if      (!strcmp(argv[i],"3d"))   o.mode = MODE_3D;
            else if (!strcmp(argv[i],"1d-x")) o.mode = MODE_1D_X;
            else if (!strcmp(argv[i],"1d-y")) o.mode = MODE_1D_Y;
            else if (!strcmp(argv[i],"1d-z")) o.mode = MODE_1D_Z;
            else { fprintf(stderr,"Unknown mode: %s\n", argv[i]); exit(1); }
        }
        else { fprintf(stderr,"Unknown option: %s\n", argv[i]); usage(argv[0]); }
    }
    if (o.input[0] == '\0') { fprintf(stderr,"--input is required\n"); usage(argv[0]); }
    return o;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    Opts o = parse_args(argc, argv);

    /* Set OpenMP thread count before any parallel region */
    if (o.nthreads > 0) {
#ifdef _OPENMP
        omp_set_num_threads(o.nthreads);
#else
        fprintf(stderr, "Warning: --nthreads ignored (compiled without OpenMP)\n");
#endif
    }

    int nthreads_actual = 1;
#ifdef _OPENMP
    nthreads_actual = omp_get_max_threads();
#endif

    printf("\n=== compute_xi  |  threads=%d ===\n\n", nthreads_actual);

    /* ── Step 1: read HDF5 header ────────────────────────────────────────── */
    double    boxsize = 0, redshift = 0;
    long long N_total = 0;

    printf("Reading header from %s ...\n", o.input);
    if (read_header(o.input, o.ptype, &boxsize, &redshift, &N_total) != 0)
        return 1;
    printf("  PartType%d: N=%lld, BoxSize=%.4f Mpc, z=%.4f\n",
           o.ptype, N_total, boxsize, redshift);

    /*
     * The FFT path runs only when all three axes are periodic; otherwise
     * the cyclic autocorrelation does not equal the true (linear) one.
     */
    int fft_active = o.use_fft &&
                     o.periodic == (PERIODIC_X | PERIODIC_Y | PERIODIC_Z);

    /*
     * Auto Ngrid: match the particle-per-dimension count (cbrt(N)) so each
     * grid cell holds exactly one particle on average, preserving the
     * sub-Poissonian shot-noise properties of the perturbed lattice.
     * The direct path scales as Ngrid³ × N_lags, which can be very slow for
     * Ngrid ≥ 256 — use --no-fft only with a small --Ngrid in that case.
     */
    if (o.Ngrid <= 0) {
        o.Ngrid = (int)round(cbrt((double)N_total));
        printf("  Auto Ngrid = %d  (cbrt(%lld))\n", o.Ngrid, N_total);
    }

    int    Ngrid = o.Ngrid;
    double cell  = boxsize / Ngrid;  /* grid cell side length in Mpc */

    /* Set deferred defaults (need boxsize and cell) */
    if (o.rmin < 0) o.rmin = cell;           /* ≥ 1 cell ensures lags are resolved */
    if (o.rmax < 0) o.rmax = boxsize / 2.0;  /* L/2: Nyquist radial distance on a torus; PBC is exact for CIC grid */
    if (o.logbin && o.rmin <= 0.0) {
        printf("WARNING: --logbin requires rmin > 0; setting rmin = cell = %.6f\n", cell);
        o.rmin = cell;
    }

    printf("  Ngrid=%d, cell=%.4f Mpc  (mean occupancy=%.1f particles/cell)\n",
           Ngrid, cell, (double)N_total / (double)((long)Ngrid*Ngrid*Ngrid));
    printf("  r range: [%.4f, %.4f] Mpc  (%s bins)\n",
           o.rmin, o.rmax, o.logbin ? "log" : "linear");

    /* ── Step 2: read all particle coordinates ───────────────────────────── */
    printf("Reading %lld particles ...\n", N_total);
    double *coords = NULL;
    if (read_coords(o.input, o.ptype, &coords, N_total) != 0)
        return 1;
    printf("  %.2f GB loaded\n",
           (double)N_total * 3 * sizeof(double) / (1024.0*1024.0*1024.0));

    /* ── Step 3: build CIC density grid ─────────────────────────────────── */
    printf("Building %s density grid (%d^3 = %ld cells) ...\n",
           order_name(o.order), Ngrid, (long)Ngrid*Ngrid*Ngrid);
    long   Ngrid3 = (long)Ngrid * Ngrid * Ngrid;
    double *den   = (double *)calloc((size_t)Ngrid3, sizeof(double));
    if (!den) { fprintf(stderr, "OOM for density grid\n"); return 1; }

    mas_assign(coords, N_total, boxsize, Ngrid, o.order, 0.0, den);

    /*
     * Normalise: den[i] = count[i] / mean_count = 1 + δ[i].
     * mean_count = N_total / Ngrid³ (exact for integer-multiple grids).
     * Track min/max as a sanity check (a perfect lattice → den near 1 at high z).
     *
     * IMPORTANT: if --vel is requested we need the raw count grid (before
     * normalisation) as the denominator for the mass-weighted velocity mean.
     * We save it to count_raw before dividing.
     */
    double mean = 0.0;
    for (long i = 0; i < Ngrid3; i++) mean += den[i];
    mean /= (double)Ngrid3;

    /*
     * Save raw counts for the velocity CIC assignment (needed as denominator
     * in mass-weighted mean: v_cell = Σ w·v / Σ w = vsum / count_raw).
     * Allocate only if --vel is requested.
     */
    double *count_raw = NULL;
    if (o.compute_vel) {
        count_raw = (double *)malloc((size_t)Ngrid3 * sizeof(double));
        if (!count_raw) { fprintf(stderr, "OOM for count_raw\n"); return 1; }
        for (long i = 0; i < Ngrid3; i++) count_raw[i] = den[i]; /* raw counts */
    }

    double dmin = DBL_MAX, dmax = -DBL_MAX;
    for (long i = 0; i < Ngrid3; i++) {
        den[i] /= mean;
        if (den[i] < dmin) dmin = den[i];
        if (den[i] > dmax) dmax = den[i];
    }
    printf("  mean=%.4f counts/cell, den in [%.4f, %.4f]\n", mean, dmin, dmax);

    /* ── Optional: P(k) estimate (consistent with compute_pk.py) ──────────
     *
     * Independent of the ξ(r) autocorrelation: builds its own interlaced,
     * deconvolved density power spectrum from the same particle coordinates.
     * Runs while coords is still resident (before the velocity step frees it).
     */
    if (o.compute_pk) {
        char pk_out_path[512];
        if (o.pk_output[0] != '\0') {
            strncpy(pk_out_path, o.pk_output, 511);
            pk_out_path[511] = '\0';
        } else {
            /* Auto-derive: substitute "xi_cic" → "pk" in the xi output path. */
            char *p = strstr(o.output, "xi_cic");
            if (p) {
                size_t prefix_len = (size_t)(p - o.output);
                snprintf(pk_out_path, sizeof(pk_out_path), "%.*s%s%s",
                         (int)prefix_len, o.output, "pk", p + strlen("xi_cic"));
            } else {
                char *dot = strrchr(o.output, '.');
                if (dot)
                    snprintf(pk_out_path, sizeof(pk_out_path), "%.*s_pk%s",
                             (int)(dot - o.output), o.output, dot);
                else
                    snprintf(pk_out_path, sizeof(pk_out_path), "%s_pk", o.output);
            }
        }
        printf("\n-- Power spectrum P(k) --\n");
        printf("Estimating P(k): interlaced %s assignment, %d k-bins, h=%.4f ...\n",
               order_name(o.order), o.nkbins, o.H0 / 100.0);
        compute_and_write_pk(coords, N_total, boxsize, Ngrid, o.order,
                             o.H0 / 100.0, o.nkbins, redshift, o.ptype,
                             o.input, order_name(o.order), pk_out_path);
    }

    /* coords still needed for velocity CIC assignment; free after that step */
    if (!o.compute_vel) {
        free(coords);
        coords = NULL;
    }

    /* ── Step 4: construct radial bin edges ─────────────────────────────── */
    int     nbins     = o.nbins;
    double *bin_edges = (double *)malloc((size_t)(nbins+1) * sizeof(double));
    double *r_c       = (double *)malloc((size_t)nbins * sizeof(double));

    if (o.logbin) {
        /*
         * Log-spaced bins: uniform spacing in log10(r).
         * Bin centre = geometric mean of edge pair = sqrt(e_lo × e_hi).
         * Preferred for ξ(r) because the dynamic range in r is large and
         * equal relative bin widths Δr/r = const give uniform signal-to-noise.
         */
        double log_lo = log10(o.rmin), log_hi = log10(o.rmax);
        for (int b = 0; b <= nbins; b++)
            bin_edges[b] = pow(10.0, log_lo + (log_hi - log_lo) * b / nbins);
        for (int b = 0; b < nbins; b++)
            r_c[b] = sqrt(bin_edges[b] * bin_edges[b+1]);
    } else {
        /* Linear bins: uniform Δr.  Centre = arithmetic mean of edges. */
        for (int b = 0; b <= nbins; b++)
            bin_edges[b] = o.rmin + (o.rmax - o.rmin) * b / nbins;
        for (int b = 0; b < nbins; b++)
            r_c[b] = 0.5 * (bin_edges[b] + bin_edges[b+1]);
    }
    printf("Bins: %d %s in [%.4f, %.4f] Mpc\n",
           nbins, o.logbin ? "log" : "linear", o.rmin, o.rmax);

    /* ── Step 5: build list of lag vectors (only needed for direct path) ─ */
    long  n_lags = 0;
    Lag  *lags   = NULL;
    if (!fft_active) {
        lags = build_lag_list(Ngrid, cell, o.rmin, o.rmax, o.mode, &n_lags);
        printf("Lag vectors: %ld  (positive-octant integer grid displacements)\n",
               n_lags);

        double ops             = (double)n_lags * (double)Ngrid3;
        double ops_per_thread  = ops / nthreads_actual;
        printf("Estimated work: %.2e cell-pair ops  (%.1e per thread)\n",
               ops, ops_per_thread);
        if (ops_per_thread > 5e11)
            printf("WARNING: heavy workload — consider --Ngrid 128 or smaller --rmax.\n");
    } else {
        printf("Using FFT autocorrelation: O(Ngrid³ log Ngrid) ~ %.2e ops\n",
               (double)Ngrid3 * log2((double)Ngrid3));
    }

    /* ── Step 6: accumulate DD, DR, RR (FFT or direct lag-sum) ────────── */
    const char *per_str =
        (o.periodic == (PERIODIC_X|PERIODIC_Y|PERIODIC_Z)) ? "xyz" :
        (o.periodic == 0)                                   ? "none" : "partial";
    printf("%s (periodic=%s) ...\n",
           fft_active ? "FFT autocorrelation" : "Pair counting", per_str);
    fflush(stdout);

    double *DD = (double *)calloc((size_t)nbins, sizeof(double));
    double *DR = (double *)calloc((size_t)nbins, sizeof(double));
    double *RR = (double *)calloc((size_t)nbins, sizeof(double));

    if (fft_active) {
        accumulate_pairs_fft(den, Ngrid, cell, o.mode,
                             bin_edges, nbins, DD, DR, RR);
    } else {
        accumulate_pairs(den, Ngrid, lags, n_lags,
                         bin_edges, nbins, DD, DR, RR, o.periodic);
        free(lags);
        lags = NULL;
    }
    free(den);

    /* ── Step 7: Landy-Szalay estimator  ξ = (DD − 2·DR + RR) / RR ─────
     *
     * After normalising by RR (the analytic random-random count):
     *   DD/RR → <den_lo × den_hi> = 1 + ξ(r)
     *   DR/RR → <den_lo>           = 1          (mean overdensity = 0)
     *   ξ = (DD − 2·DR + RR) / RR = 1+ξ − 2 + 1 = ξ  ✓
     */
    double *xi = (double *)malloc((size_t)nbins * sizeof(double));
    double xi_min = DBL_MAX, xi_max = -DBL_MAX;
    for (int b = 0; b < nbins; b++) {
        xi[b] = (RR[b] > 0.0) ? (DD[b] - 2.0*DR[b] + RR[b]) / RR[b] : 0.0/0.0;
        if (isfinite(xi[b])) {
            if (xi[b] < xi_min) xi_min = xi[b];
            if (xi[b] > xi_max) xi_max = xi[b];
        }
    }
    printf("xi in [%.4e, %.4e]\n", xi_min, xi_max);

    /* ── Step 8: write output ─────────────────────────────────────────────
     *
     * Output columns: r_avg  r_low  r_high  xi(r)  DD  DR  RR
     * Columns 0-3 match the format written by compute_xi_corrfunc.c so that
     * compute_pk.py can read both files identically (cols 1-3 for r_low,
     * r_high, xi) and overlay them on the same panel.
     * DD/DR/RR are written for diagnostics; they are not needed for plotting.
     */
    const char *mode_str =
        (o.mode==MODE_3D) ? "3d" : (o.mode==MODE_1D_X) ? "1d-x" :
        (o.mode==MODE_1D_Y) ? "1d-y" : "1d-z";

    FILE *fp = fopen(o.output, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s for writing\n", o.output); return 1; }

    fprintf(fp,
        "# Two-point correlation function  |  compute_xi.c\n"
        "# Method  : %s density-grid autocorrelation (Landy-Szalay)\n"
        "#           Equivalent to Hankel transform of FFT P(k).\n"
        "# File    : %s\n"
        "# PartType: %d   N_particles: %lld\n"
        "# Ngrid   : %d   cell: %.6f Mpc\n"
        "# Mode    : %s\n"
        "# Periodic: %s\n"
        "# BoxSize : %.6f Mpc   z: %.6f\n"
        "# Binning : %s\n"
        "# Note    : r in Mpc (SWIFT IC units, not Mpc/h)\n"
        "# Columns : r_avg  r_low  r_high  xi(r)  DD  DR  RR\n",
        order_name(o.order),
        o.input, o.ptype, N_total,
        Ngrid, cell, mode_str, per_str,
        boxsize, redshift,
        o.logbin ? "logarithmic" : "linear");

    for (int b = 0; b < nbins; b++) {
        if (o.r2_plot)
            fprintf(fp, "%12.6f  %12.6f  %12.6f  %15.8e  %15.6e  %15.6e  %15.6e  %15.8e\n",
                    r_c[b], bin_edges[b], bin_edges[b+1],
                    xi[b], DD[b], DR[b], RR[b],
                    xi[b] * r_c[b] * r_c[b]);
        else
            fprintf(fp, "%12.6f  %12.6f  %12.6f  %15.8e  %15.6e  %15.6e  %15.6e\n",
                    r_c[b], bin_edges[b], bin_edges[b+1],
                    xi[b], DD[b], DR[b], RR[b]);
    }
    fclose(fp);
    printf("Saved → %s\n\n", o.output);

    free(DD); free(DR); free(RR); free(xi);

    /* ── Optional: velocity correlation ψ(r) = ⟨v(x)·v(x+r)⟩ ─────────────
     *
     * Requires --vel flag.  Steps:
     *   (a) Read PartType{ptype}/Velocities from HDF5 (float32 → double, km/s)
     *   (b) CIC-assign mass-weighted velocity to three grids (vx, vy, vz)
     *   (c) Enumerate the same lag list (already built above)
     *   (d) Accumulate ψ(r) = Σ v(lo)·v(hi) / N_cell_pairs
     *   (e) Write to vel output file
     *
     * The lag list `lags` has already been freed above; we rebuild it here.
     * The bin_edges/r_c arrays are still valid.
     */
    if (o.compute_vel) {

        /* (a) Read velocities */
        printf("\n-- Velocity correlation ψ(r) --\n");
        printf("Reading velocities from %s ...\n", o.input);
        double *vels = NULL;
        if (read_velocities(o.input, o.ptype, &vels, N_total) != 0) {
            fprintf(stderr, "Failed to read velocities; skipping ψ(r).\n");
            free(coords); free(count_raw);
            goto cleanup;
        }

        /* Print rms velocity as a sanity check */
        double vrms2 = 0.0;
        for (long long p = 0; p < N_total; p++)
            vrms2 += vels[p*3+0]*vels[p*3+0]
                   + vels[p*3+1]*vels[p*3+1]
                   + vels[p*3+2]*vels[p*3+2];
        vrms2 /= (double)N_total;
        printf("  v_rms = %.4f km/s  (3D, particle average)\n", sqrt(vrms2));

        /* (b) CIC-assign velocity to three grids */
        printf("Building velocity CIC grids (%d^3) ...\n", Ngrid);
        double *vx_f = (double *)calloc((size_t)Ngrid3, sizeof(double));
        double *vy_f = (double *)calloc((size_t)Ngrid3, sizeof(double));
        double *vz_f = (double *)calloc((size_t)Ngrid3, sizeof(double));
        if (!vx_f || !vy_f || !vz_f) {
            fprintf(stderr, "OOM for velocity grids\n");
            free(vels); free(coords); free(count_raw);
            goto cleanup;
        }

        mas_vel_assign(coords, vels, N_total, boxsize, Ngrid, o.order,
                       count_raw, vx_f, vy_f, vz_f);
        free(vels); free(coords); free(count_raw);
        coords = NULL;

        /* (c,d) Accumulate ψ(r) — FFT fast path or direct */
        double *VV     = (double *)calloc((size_t)nbins, sizeof(double));
        double *RR_vel = (double *)calloc((size_t)nbins, sizeof(double));
        if (!VV || !RR_vel) {
            fprintf(stderr, "OOM for VV/RR_vel\n");
            free(vx_f); free(vy_f); free(vz_f);
            goto cleanup;
        }

        if (fft_active) {
            printf("FFT autocorrelation (velocity) ...\n"); fflush(stdout);
            accumulate_vel_pairs_fft(vx_f, vy_f, vz_f, Ngrid, cell, o.mode,
                                     bin_edges, nbins, VV, RR_vel);
        } else {
            long n_lags_v = 0;
            Lag *lags_v = build_lag_list(Ngrid, cell, o.rmin, o.rmax,
                                         o.mode, &n_lags_v);
            printf("Lag vectors (velocity): %ld\n", n_lags_v);
            printf("Pair counting (velocity) ...\n"); fflush(stdout);
            accumulate_vel_pairs(vx_f, vy_f, vz_f, Ngrid,
                                 lags_v, n_lags_v,
                                 bin_edges, nbins, VV, RR_vel, o.periodic);
            free(lags_v);
        }
        free(vx_f); free(vy_f); free(vz_f);

        /* ψ(r) = VV / RR_vel  (normalised per cell pair) */
        double *psi = (double *)malloc((size_t)nbins * sizeof(double));
        if (!psi) { fprintf(stderr, "OOM for psi\n"); free(VV); free(RR_vel); goto cleanup; }

        double psi_min = DBL_MAX, psi_max = -DBL_MAX;
        for (int b = 0; b < nbins; b++) {
            psi[b] = (RR_vel[b] > 0.0) ? VV[b] / RR_vel[b] : 0.0/0.0;
            if (isfinite(psi[b])) {
                if (psi[b] < psi_min) psi_min = psi[b];
                if (psi[b] > psi_max) psi_max = psi[b];
            }
        }
        printf("psi in [%.4e, %.4e] (km/s)^2\n", psi_min, psi_max);

        /* (e) Write velocity correlation output */
        /*
         * Auto-derive vel output filename: replace "xi_cic" with "vel_cic"
         * in the xi output path, or append "_vel" before the extension.
         */
        char vel_out_path[512];
        if (o.vel_output[0] != '\0') {
            strncpy(vel_out_path, o.vel_output, 511);
        } else {
            /* Try to substitute "xi_cic" → "vel_cic" in the output filename */
            char *p = strstr(o.output, "xi_cic");
            if (p) {
                /* Copy the part before "xi_cic", then "vel_cic", then the rest */
                size_t prefix_len = (size_t)(p - o.output);
                snprintf(vel_out_path, sizeof(vel_out_path),
                         "%.*s%s%s",
                         (int)prefix_len, o.output,
                         "vel_cic",
                         p + strlen("xi_cic"));
            } else {
                /* Fallback: insert "_vel" before the last "." */
                char *dot = strrchr(o.output, '.');
                if (dot) {
                    snprintf(vel_out_path, sizeof(vel_out_path),
                             "%.*s_vel%s",
                             (int)(dot - o.output), o.output, dot);
                } else {
                    snprintf(vel_out_path, sizeof(vel_out_path),
                             "%s_vel", o.output);
                }
            }
        }

        FILE *fv = fopen(vel_out_path, "w");
        if (!fv) {
            fprintf(stderr, "Cannot open %s for writing\n", vel_out_path);
        } else {
            fprintf(fv,
                "# Velocity autocorrelation  ψ(r) = <v(x)·v(x+r)>  |  compute_xi.c\n"
                "# Method  : %s mass-weighted velocity field + lag autocorrelation\n"
                "#           ψ(r) = Σ_cells v(x)·v(x+r) / N_cell_pairs\n"
                "# Linear theory: ψ(r) = [H(z)f(z)]^2/(2π^2) ∫ P(k) j0(kr) dk\n"
                "#           where f ≈ Ω_m(z)^0.55 (Linder 2005)\n"
                "# File    : %s\n"
                "# PartType: %d   N_particles: %lld\n"
                "# Ngrid   : %d   cell: %.6f Mpc\n"
                "# Mode    : %s\n"
                "# Periodic: %s\n"
                "# BoxSize : %.6f Mpc   z: %.6f\n"
                "# Binning : %s\n"
                "# Units   : r in Mpc, ψ in (km/s)^2\n"
                "# Columns : r_avg  r_low  r_high  psi(r)\n",
                order_name(o.order),
                o.input, o.ptype, N_total,
                Ngrid, cell, mode_str, per_str,
                boxsize, redshift,
                o.logbin ? "logarithmic" : "linear");

            for (int b = 0; b < nbins; b++) {
                fprintf(fv, "%12.6f  %12.6f  %12.6f  %15.8e\n",
                        r_c[b], bin_edges[b], bin_edges[b+1], psi[b]);
            }
            fclose(fv);
            printf("Saved → %s\n", vel_out_path);
        }

        free(VV); free(RR_vel); free(psi);
    } else {
        /* --vel not requested: free coords (may have been kept above) */
        free(coords);
        coords = NULL;
    }

cleanup:
    free(bin_edges); free(r_c);
    return 0;
}
