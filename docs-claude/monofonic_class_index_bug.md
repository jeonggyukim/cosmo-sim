# monofonIC reads `index_tp_theta_m` uninitialised

Date: 2026-09-04.

monofonIC crashes at random during start-up whenever `transfer = CLASS`. The
crash rate depends on the grid size and on how many processes run at once, which
makes it look like a machine or filesystem problem. It is neither. It is an
out-of-bounds read of the CLASS source array, caused by using a source index
that CLASS never assigned.

This repository runs `transfer = CLASS`, so every run is affected, and
`scripts/ic_search/` already carries a retry loop built for these crashes with
the cause recorded incorrectly.

## 1. The mechanism

CLASS assigns each source index `index_tp_X` only when the matching flag
`has_source_X` is true. The macro that does the assigning leaves the member
untouched otherwise (`class-src/include/common.h:269`):

```c
#define class_define_index(index, condition, running_index, number_of_indices) { \
    if (condition) {                                                             \
      index = running_index;                                                     \
      running_index += number_of_indices;                                        \
    }                                                                            \
}
```

So an index whose flag was never set holds whatever was in memory.
`transfer_CLASS.cc` requests source functions by index without ever checking the
flags. With the `output` string monofonIC gives CLASS, `has_source_theta_m` is
never set, so `index_tp_theta_m` is read while uninitialised.

Sometimes the stale value lands inside the source array and garbage is read
silently. Otherwise `perturbations_sources_at_tau()` indexes past the end and
the process dies with `SIGSEGV`, during `ic_generator::initialise()`, before any
grid work.

## 2. Evidence in the sources

Read against `monofonic_build/_deps/class-src` (CLASS 3.3.3, fetched by
`external/class.cmake`) and `../monofonIC/src/plugins/`.

| Fact | Where |
|---|---|
| monofonIC asks CLASS for `output = "dTk,vTk,mTk,mPk"` | `transfer_CLASS.cc:94` |
| every source request is unguarded; the string `has_source` does not appear in the file | `transfer_CLASS.cc:422–455` |
| `has_source_theta_m` is set true in exactly one place, inside `if (has_cl_number_count)` then `if (has_nc_rsd)` | `perturbations.c:1345` |
| `has_source_delta_m` is set true in three places, which is why the density side has always worked | `perturbations.c:1275`, `:1285`, `:1342` |
| `index_tp_theta_m` is defined only under that flag | `perturbations.c:1407` |
| the velocity-transfer branch defines theta for individual species only — `theta_ur`, `theta_idr`, `theta_dr`, `theta_ncdm`, no combined species | `perturbations.c:1330–1344` |
| the struct is a plain member and nothing memsets it, so its indices are indeterminate | `transfer_CLASS.cc:55` |
| the destination vectors are zero-filled, so a fix may rely on that | `transfer_CLASS.cc:409–416` |

The same exposure applies to `index_tp_theta_cb`, `index_tp_theta_tot`,
`index_tp_delta_cb` and `index_tp_delta_tot`, each guarded by a flag set in one
place, and to `index_tp_delta_ncdm1` and `index_tp_theta_ncdm1`, whose flags are
not set at all when `N_ncdm = 0`.

**CLASS is not at fault.** It honours its own contract: `transfer.c:679` and
`:784` both test `has_source_theta_m == _TRUE_` before comparing against
`index_tp_theta_m`, and `perturbations.c:7924` writes the source only inside
code that runs when the source exists. The bug belongs to monofonIC and should
be reported there.

One robustness remark is worth passing to CLASS as a suggestion rather than a
defect. Because `class_define_index` leaves the member indeterminate rather than
setting a sentinel such as `-1`, a misuse becomes a crash whose probability
depends on the state of the allocator. An `else index = -1;` would make any such
misuse fail identically on every run and every machine.

## 3. The retry loop here was built for this, and blames the wrong cause

`pencil_seed_sweep.py:130` sets `--nretry` to 6 by default. The comment there
and at line 279 records the cause as filesystem contention:

> Hundreds of tasks starting at once contend for the CLASS and HyRec data files,
> and CLASS segfaults on some of those reads. Measured skip rate rose from 0.5%
> at 4 concurrent processes to 4.4% at 500.

That explanation is wrong. `collect_missing.py` exists to find and rerun the
seeds lost this way.

The recorded skip rates are consistent with the uninitialised index, and give an
independent estimate of the crash rate. They are measured *after* six attempts,
and each retry waits a random interval, which changes the allocator state and
makes the attempts close to independent. Backing out the per-attempt probability
from a skip rate `s` as `s^(1/6)`:

| skip rate recorded here | concurrent processes | implied per-attempt crash rate |
|---|---|---|
| 0.5% | 4 | 41% |
| 4.4% | 500 | 59% |

Reported elsewhere, on a different machine and configuration: 70% at
`GridRes = 32` and about 12% at `GridRes = 256`. This repository runs
`GridRes = 128`, and the implied 41–59% falls between those two values. The
dependence on concurrency is real but is a second-order effect on allocator
state, not filesystem contention.

## 4. Consequences

**Existing results are unaffected.** The buffer that receives the out-of-bounds
read is `t_tot` under `MATTER_`, the combined matter velocity. Nothing consumes
it: `theta_matter` appears only in the enumeration at
`include/transfer_function_plugin.hh:31` and in the CAMB file plugin. Particle
velocities come from the Lagrangian perturbation theory potentials, not from
this array. The reasoning recorded in `collect_missing.py` — that a task dying
is unrelated to the fields it was carrying, so losses do not select on the
realisation — still holds.

**Compute is being wasted.** At a per-attempt crash rate near 50%, roughly half
of all monofonIC invocations in a sweep die and are repeated.

## 5. The fix

Request a source only once CLASS reports it has produced it, guarding each call
with its `has_source_*` flag and letting the zero-filled destination vectors
stand in for the missing ones. All eleven flag names exist in
`class-src/include/perturbations.h`. Two amendments are worth making on top.

**Reconstruct `theta_m` rather than leaving it zero.** monofonIC runs in
synchronous gauge (`transfer_CLASS.cc:100`), in which the cold dark matter
velocity is zero by construction — which is why `index_tp_theta_cdm` is never
requested anywhere in the file. The combined matter velocity therefore reduces
to

    theta_m = (rho_b * theta_b + rho_ncdm * theta_ncdm) / rho_m

with no cold dark matter term, and both densities are already in the background
vector read a few lines above. That is cheaper than the alternative of enabling
the number-count angular power spectra in CLASS purely to make it produce one
source function.

**Fail loudly rather than silently.** With the guards in place, a request for
`theta_matter` returns real values under `transfer = CAMB` and zeros under
`transfer = CLASS`: two plugins implementing one interface, disagreeing with no
warning. Since nothing consumes `theta_matter` today, the CLASS plugin should
raise an error when asked for a source CLASS did not produce, so the next use
gets a message instead of a wrong answer. The same applies to the neutrino
sources when `N_ncdm = 0`, and to `index_tp_h_prime` and `index_tp_eta_prime`,
which are needed for the gauge handling.

## 6. What to do here

1. Report the bug upstream to monofonIC.
2. Apply the fix to the fork this repository uses, and measure the crash rate at
   `GridRes = 128` before and after.
3. Once the fix is in, correct the comments at `pencil_seed_sweep.py:130` and
   `:279`, and reconsider whether `--nretry` and `collect_missing.py` are still
   needed.
