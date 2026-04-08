# Corrfunc: Computing ξ(r) for IC Particles

## What is Corrfunc?

[Corrfunc](https://github.com/manodeep/Corrfunc) (Sinha & Garrison 2020, MNRAS 491) is a
high-performance pair-counting library for computing 2-point statistics on CPUs.
It uses AVX/SSE SIMD intrinsics and a cell-linked-list spatial decomposition to
count pairs in O(N × n_neighbors) time, typically 10–100× faster than naive O(N²)
implementations.

For a periodic box (no randoms needed), Corrfunc implements the **natural estimator**
(Peebles & Hauser 1974):

$$\xi(r) = \frac{DD(r)}{DD_\mathrm{rand}(r)} - 1$$

where DD(r) is the actual data–data pair count in shell [r, r+dr), and
DD_rand = N(N−1)/2 × V_shell/V_box is the analytically expected count for a
spatially uniform distribution. This is exact for periodic geometry and requires
no random catalogue.

## Overview

`compute_xi.c` computes the real-space 2-point correlation function ξ(r) for
SWIFT IC particle data using the [Corrfunc](https://github.com/manodeep/Corrfunc)
C static library.

## Build

Corrfunc must be built from source first:

```bash
./prepare-corrfunc.sh    # clones + builds into ~/Dropbox/Projects/Corrfunc (jgkim/Darwin)
make compute_xi          # builds compute_xi binary
```

## Usage

```bash
./compute_xi <ics.hdf5> <binfile> [nthreads] [-n NSUB]
```

| Argument | Description |
|----------|-------------|
| `ics.hdf5` | SWIFT IC file (reads `PartType1/Coordinates`) |
| `binfile` | ASCII file with two columns: `rmin rmax` per bin |
| `nthreads` | Number of OpenMP threads (default: 4) |
| `-n NSUB` | Subsample to NSUB particles (auto if omitted) |

Output goes to stdout: `r_avg  r_low  r_high  xi  npairs`

### Generate r-bins

Always generate bins appropriate for each IC file using `make_rbins.py`:

```bash
python make_rbins.py --hdf5 ics_swift_n256_z127_L25.hdf5   # reads BoxSize and N automatically
python make_rbins.py -N 256 -L 25 --H0 67.11               # or specify directly
```

**Do not reuse `rbins.txt` across runs with different box sizes or resolutions.**
Bins are in **Mpc** (matching SWIFT HDF5 coordinate units, which store positions in Mpc not Mpc/h).

### Bin parameter choices (literature-validated)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `rmin` | 2 × mean spacing | Below the mean interparticle spacing (d = L/N) ξ = −1 by construction; 2× is a safe conservative choice (1–3× used in practice) |
| `rmax` | L/3 | Hard limit is L/2 (periodic images); L/3 is the standard conservative choice, more commonly used than L/4 in recent simulations |
| `nbins` | 24 log-spaced | Gives bin size ~0.07 in log₁₀, below the TreeCorr-recommended ≤ 0.1; 15–30 bins is typical |
| `DD_target` | 10,000 pairs | Targets σ(ξ) ≈ 1% Poisson error in the smallest bin; standard for IC validation |
| Spacing | Logarithmic | Standard for ξ(r) (Hamilton 1993; Corrfunc, TreeCorr defaults) |

References: Peebles & Hauser (1974); Hamilton (1993); Sinha & Garrison (2020, Corrfunc); UNIT simulations (Chuang et al. 2019); TreeCorr docs.

## Subsampling

Pair counting is O(N × n_neighbors) where n_neighbors ∝ rmax³. For large N
and rmax (e.g. 16M particles, rmax=10 Mpc/h), this is intractable. Random
subsampling is statistically valid for a periodic box because the Corrfunc
estimator normalises by N_s(N_s−1) — no bias is introduced.

### Rule of thumb

The statistical error on ξ(r) is dominated by Poisson shot noise on pair counts:

$$\sigma(\xi) \approx \frac{1+\xi}{\sqrt{DD}} \approx \frac{1}{\sqrt{DD}} \quad [\xi \approx 0 \text{ for ICs}]$$

Target **DD ≥ 10,000 pairs** in the smallest bin → ~1% precision.

For a periodic box with volume V, the expected pair count in the smallest bin
(shell volume V_shell = 4π/3 × (r_hi³ − r_lo³)) is:

$$DD = \frac{N_s(N_s-1)}{2} \cdot \frac{V_\mathrm{shell}}{V}$$

Solving for N_s:

$$N_s = \left\lceil \sqrt{\frac{2 \cdot DD_\mathrm{target} \cdot V}{V_\mathrm{shell,min}}} \right\rceil$$

This is computed automatically when `-n` is not specified.

### Practical numbers (L=37.25 Mpc/h, rmin=0.1 Mpc/h, rmax=10 Mpc/h)

| Resolution | N_total | N_auto | Runtime |
|------------|---------|--------|---------|
| 64³        | 262K    | 262K (all) | ~7s |
| 128³       | 2.1M    | 563K   | ~30s   |
| 256³       | 16.8M   | 563K   | ~30s   |

The auto N_s is independent of N_total once N_total >> N_s — it only depends
on the box volume and bin edges.

### Reference

- Peebles & Hauser (1974) — pair counting statistics
- Hamilton (1993) — ξ(r) estimator variance
- Angulo & White (2010) — subsampling effects in N-body simulations

## compute_xi.c — Code Structure

```
main()
├── parse args (hdf5file, binfile, nthreads, -n NSUB)
├── read_positions()         — open SWIFT HDF5, read BoxSize + PartType1/Coordinates
│     SWIFT stores coordinates in Mpc (not Mpc/h); BoxSize likewise in Mpc.
│     Float32 in file → promoted to double for Corrfunc.
├── read_first_bin()         — parse first line of binfile to get V_shell_min
├── auto-size Nsub           — solve N_s² V_shell / (2 V_box) = DD_TARGET
├── subsample()              — Fisher-Yates partial shuffle, seed=42 (reproducible)
├── countpairs_xi()          — Corrfunc periodic natural estimator
│     options.periodic = 1  : uses periodic boundary conditions
│     options.float_type = 8: double precision internally
│     options.verbose = 1   : prints per-bin progress to stderr
└── print results to stdout  — r_avg  r_low  r_high  xi  npairs
```

### Key data flow

1. **HDF5 → positions**: `read_positions()` opens the file with HDF5 C API,
   reads the `(N, 3)` float32 coordinate array from `PartType1/Coordinates`,
   and deinterleaves it into separate `X`, `Y`, `Z` double arrays.

2. **Auto-subsampling**: The innermost bin has the smallest shell volume V_shell_min,
   so it sets the minimum Nsub. The formula targets DD_TARGET = 10,000 pairs there,
   giving σ(ξ) ≈ 1%. If the total particle count N is smaller than the target Nsub,
   all particles are used.

3. **Fisher-Yates shuffle**: A partial in-place shuffle selects Nsub particles
   without replacement (fixed seed=42). Only the first Nsub elements are passed
   to Corrfunc.

4. **Corrfunc call**: `countpairs_xi()` takes the Nsub positions, boxsize, nthreads,
   and binfile path. It returns a `results_countpairs_xi` struct containing
   `ravg`, `rupp`, `xi`, and `npairs` arrays indexed 0..nbin.

5. **Output**: `rupp[0]` is the lower edge of the first real bin. The loop
   reconstructs `rlow` by carrying `rupp[i-1]` forward. Note: `ravg` is zero
   unless `options.need_avg_sep = 1` is set (not currently enabled).

### Known limitations

- **Units**: SWIFT HDF5 stores coordinates in **Mpc** (not Mpc/h). The bin file
  must be in the same units (Mpc). Use `make_rbins.py` which handles this correctly.
- **`r_avg` is always 0**: Corrfunc only computes average separations when
  `options.need_avg_sep = 1`, which adds overhead. Currently disabled; use
  bin midpoints for plotting.

## When ξ(r) is the wrong tool

At high redshift (z ≳ 10), IC particles are nearly on a perfect lattice. The
measured ξ(r) is dominated by lattice artifacts at small scales and shot noise
at large scales; the cosmological signal (~10⁻⁶ at z=45) is undetectable with
a subsampled pair count. Use **P(k) via FFT** (`compute_pk.py`) for IC validation
instead — it uses all particles in O(N log N) and has much better S/N.

ξ(r) from particles is only practical at low z (z ≲ 2) where ξ ~ 1 at small
scales and the subsampling target of DD ~ 10,000 gives useful signal.

## Key References

- **Peebles & Hauser (1974)** ApJS 28, 19 — natural estimator ξ = DD/RR − 1
- **Hamilton (1993)** ApJ 417, 19 — variance of ξ(r): σ(ξ) ≈ 1/√DD
- **Sinha & Garrison (2020)** MNRAS 491, 3022 — Corrfunc: SIMD pair counting, periodic estimator (**primary reference**)
- **Jing (2005)** ApJ 620, 559 — CIC window correction for P(k)
- **Sefusatti et al. (2016)** MNRAS 460, 3624 — interlaced CIC for near-aliasing-free P(k)
- **Hahn & Abel (2011)** MNRAS 415, 2101 — IC validation via P(k) (Fig. 4); transfer function ξ(r) (Fig. 2)
- **Feldman, Kaiser & Peacock (1994)** ApJ 426, 23 — FKP estimator; shot noise vs. cosmic variance trade-off

## Performance Notes

- `rmax` should be ≤ L/3 for the cell-linked-list to be efficient; near L/2 the
  algorithm approaches O(N²) as almost every cell is a neighbor
- Corrfunc uses AVX/SSE SIMD and is typically 10–100× faster than naive pair counting
- On macOS, must compile with `clang -fopenmp=libomp` to match the Corrfunc build
