# GPU porting of SWIFT gravity + SPH: survey and plan

Date: 2026-07-24. Revised 2026-07-27 after reading pkdgrav3 and cuGOTPM at source
level; revised again 2026-07-28 after Ivkovic 2026, Potter 2017, and inspecting the
fork's live branches. Context: KIAS project to produce a SWIFT build with
both hydro and (at least short-range) gravity on the GPU. An AI coding agent is
expected to do much of the mechanical porting work.

**Why GH200 specifically**: Hangang, Korea's forthcoming supercomputer, will be
built on Grace Hopper. The assessment is judged against that machine, not GPUs in
general, and several conclusions depend on features unique to it — the coherent
~900 GB/s NVLink-C2C link between CPU and GPU, and the ARM (Neoverse V2) host
CPU. The ARM host is why SWIFT's missing NEON/SVE path in `src/vector.h` matters
here and would not matter on an x86 GPU node.

Companion artifact (plain-English write-up for collaborators):
https://claude.ai/code/artifact/0043020a-8873-4fc0-b419-ac16edaa08b0
Republish by passing that URL as `url` to the Artifact tool; do not mint a new one.

## Code locations (local checkouts, siblings of this repo)

| Path | What | State examined |
|------|------|----------------|
| `../SWIFT` | Upstream SWIFT (CPU), branch `test-star-formation` | full architecture scan |
| `../SWIFT-GPU` | Fork `abouzied-nasar/SWIFT`, branch `gpu_master` @ d1b3b5a1 | GPU layer read in detail |
| `../pkdgrav3` | pkdgrav3 3.5 @ dcce745f | gravity + CUDA path read in detail |
| `../cuGOTPM-CPL` | `MinseongAstro/cuGOTPM-CPL`, branch `cuGOTPM` @ ede8929 | CUDA path read in detail |

Papers:
- Nasar et al. 2026, RASTI, arXiv:2505.14538 — the SWIFT-GPU fork's paper
  (`~/Documents/ref_SWIFT/Nasar-2026-SWIFT-heterogeneous-GPU-RASTI.pdf`)
- Meier, Potter, Reinhardt & Stadel 2026, ApJ 1000:266 — SPH in pkdgrav3
  (`~/Dropbox/Papers/Papers-all/Meier-2026-Smoothed-Particle-Hydrodynamics-in-pkdgrav3-*.pdf`)
- Dubinski, Kim, Park & Humble 2003, New Astronomy 9, 111 — GOTPM
- Potter, Stadel & Teyssier 2017, ComAC 4, 2 — pkdgrav3 method paper; §3.3 states the
  CPU/GPU split rationale and the runtime flop/byte dispatch
  (`~/Dropbox/Papers/Papers-all/Potter-2017-PKDGRAV3-*.pdf`)
- Ivkovic et al. 2026, arXiv:2606.23891 — memory layouts for GPU data-transfer buffering;
  measures pack cost at 60–80% on GH200 vs ~2% on PCIe, and reports a **negative result**
  on reading host memory in place (`~/Documents/ref_SWIFT/Ivkovic-2026-*.pdf`).
  Public mini-app: github.com/mladenivkovic/swiftgpupacksim

## Bottom line

**SWIFT stays the base. Port neither pkdgrav3 nor cuGOTPM. Write SWIFT's own
short-range P2P offload, taking pkdgrav3's threading design and cuGOTPM's
CPU-fallback slot pool.**

**Branch from upstream SWIFT, not from the SWIFT-GPU fork** (decided 2026-07-28).
The fork has no gravity, so there is nothing to inherit for our actual task. What
building on it would save is ~2,750 lines of pack/offload machinery, all shaped
around the SPH calculation and all needing a rewrite for gravity anyway. Against
that: four upstream files are forked *copies* needing manual merges forever, a HIP
backend that would pressure any gravity kernel to exist twice, and two moving
targets to track instead of one. Read the fork closely; branch from upstream.

~3,000–5,000 lines, 8–12 person-months to a validated prototype, 18+ to production.
Line counts of *existing* code are measured; these projections are not, and are the
numbers most likely to be wrong. Calibration: the fork's authors needed a funded
multi-year effort for three hydro loops and no gravity.

**Expected gain, added 2026-07-30 — decide whether it justifies the effort before
starting.** On our own 25 Mpc/h run, the distribution of active particles per step
caps a gravity offload at 1.6–2.0x whole-run before any pack cost is counted, and the
SPH-side evidence bounds any "offload from CPU SWIFT" strategy below roughly 2.4x.
Both sections are below. The one number that would change this picture — the
short-range P2P share of a step on our own problem — is not yet measured.

## Measured line counts (2026-07-27, raw `wc -l`)

| Component | SWIFT | pkdgrav3 3.5 | cuGOTPM |
|---|---|---|---|
| Source total | 301,830 (.c 126,904 + .h 174,926) | 107,703 (excl. vendored blitz/, fmt/) | ~97,100 |
| Gravity core | 10,444 | 8,795 (`gravity/`) | ~6,800 |
| CUDA in gravity path | 0 | 1,047 | 6,805 (many near-duplicate variants) |
| GPU pack/staging | 0 | 925 (`gpu/`) | — |
| Galaxy-formation subgrid | 77,610 | 1,826 | 0 |
| — EAGLE-specific | 33,509 | — | — |
| License | LGPLv3 | GPLv3 | **none declared** |

pkdgrav3's 1,826 lines of SF/BH physics live under `meshless/` (MFM), *not*
under the new `SPH/` path. The GPU-accelerated SPH path has no subgrid at all.
This is why the port direction is engine-design-into-SWIFT, not physics-into-pkdgrav3.

## SWIFT-GPU fork: what exists

- Offloads only the three SPHENIX pair loops (density, gradient, force). Task
  scheduler, tree, ghost/kick/drift, **all of gravity**: CPU.
- GPU-touched files total 19,414 lines, but `engine_maketasks_gpu.c` is a *copy*
  of upstream `engine_maketasks.c` (4,303 vs 4,300) with only **123 changed
  lines**. `cell_pack.c` differs by 11, `runner_main_cuda.c` by ~150. Genuinely
  new: `src/cuda/` (1,839), pack metadata+params (446), pack wrappers (466),
  trimmed hydro interaction header (1,005).
- **Duplication is a liability, not a saving.** Four upstream files are forked
  copies, so every upstream change must be re-applied by hand. Plus a full HIP
  backend (~8,700 lines) re-implementing the CUDA path.
- Extension points for a gravity subtype: `task_subtype_gpu_density/gradient/force`
  and `enum gpu_task_type` in `src/task.h`.

## Nasar 2026 performance numbers, corrected 2026-07-30

The earlier summary in this file — "3.5–7.5x on offloaded kernels, 1.8x full-sim" —
was wrong twice, and the wrong version reached the collaborator artifact.

- **3.5 and 7.5 are not a range.** Abstract, verbatim: "up to ∼3.5 and ∼7.5 speedups
  for the offloaded computations when *including* and *excluding* the time required
  to prepare and post-process data transfers on the CPU side, respectively." One
  measurement quoted twice. The difference between the two is the pack/unpack cost.
- **The 1.8x does not transfer to a run with gravity.** The benchmark is the 3-D
  Gresho–Chan vortex at 256³, which contains no gravity. In the paper's own test
  descriptions the word "gravity" does not appear. The three offloaded SPH loops are
  1903 of 2032 ms, **94% of the step**.

Measured breakdown (§4.1, §4.2, Table 4), one step, 256³, one GH200:

| | CPU, 72 Grace threads | With H100 |
|---|---|---|
| Whole step | 2032 ms | 1132 ms |
| The three SPH loops | 1903 ms | 929 ms |
| Everything else, plus time outside task execution | 129 ms | 203 ms |

- Whole step 2032/1132 = **1.8x**. The SPH loops alone 1903/929 = **2.05x**.
- Coverage was never their limit. With 94% offloaded, a free GPU would have given
  ~16x.
- Of the 929 ms, **~80% is CPU-side pack/unpack** (~743 ms), leaving ~186 ms for
  everything else including the kernels. Physics against the CPU's 1903 ms is ~10x.
  Removing the pack entirely puts the step near 389 ms, **~5.2x**.
- Time outside task execution rose 129 → 203 ms. Their §4.2: "the tasks dependent on
  these N leaf computations must wait until the results are copied back onto the CPU
  memory before they are unlocked and then enqueued." Batching for the GPU delays the
  dependency releases that CPU tasking performs as each task finishes.
- Their fork applied unchanged to a run where SPH is ~40% of the step gives
  1/(0.60 + 0.40/2.05) = **~1.26x**.

**Upper bound on the offload strategy itself.** Nasar §4.2 cites SHAMROCK
(David-Cléris 2025), which holds all particle data on the GPU and does no CPU task
scheduling: 20M updates/s on the same H100, against SWIFT's 8.26M CPU baseline,
2.4x. Their offload reaches 14.8M, 1.8x. So "keep CPU SWIFT, offload pieces" is
bounded below roughly 2.4x for SPH and they are most of the way to it. Not a
controlled comparison — different code, different test problem — but it bounds the
ambition. Gravity's higher flop/byte is the reason to expect better than SPH. That
expectation is what this project would test, and nobody has tested it.

## Coverage ceiling on our own problem, measured 2026-07-30

Source: `~/Documents/timesteps.txt`, a SWIFT `timesteps.txt` from a 25 Mpc/h
uniresolution run — 256³ gas + 256³ DM, EAGLE-XL subgrid, SPHENIX with Wendland C2,
4 MPI ranks × 16 threads, z = 127 → 0. 220,577 usable steps, 53.17 h wall, 9.32 h
(17.5%) dead time. Five lines spliced by concurrent MPI writes were dropped.

**The median step updates 18 gas particles out of 16,777,216.**

| active gravity particles | steps | share of steps | share of wall |
|---|---|---|---|
| < 10 | 89,765 | 40.7% | 14.3% |
| 10 – 100 | 58,254 | 26.4% | 9.9% |
| 100 – 10³ | 34,216 | 15.5% | 6.9% |
| 10³ – 10⁴ | 20,068 | 9.1% | 6.0% |
| 10⁴ – 10⁵ | 9,637 | 4.4% | 5.7% |
| 10⁵ – 10⁶ | 4,768 | 2.2% | 8.9% |
| 10⁶ – 10⁷ | 2,660 | 1.2% | 13.5% |
| > 10⁷ | 1,209 | 0.55% | 34.8% |

**42.8% of the wall clock is in steps updating fewer than 10⁵ particles**, below the
work needed to pay for a handover. Coverage and Amdahl against the minimum active
count at which offloading is worth doing:

| offload threshold | max coverage f | ceiling 1/(1−f) | with 5x on f |
|---|---|---|---|
| 10⁴ (optimistic) | 63.0% | 2.70x | 2.01x |
| 10⁵ (reasonable) | 57.2% | 2.34x | 1.84x |
| 10⁶ (conservative) | 48.3% | 1.94x | 1.63x |

Upper bounds, not predictions: inside a large step only part of the time is
short-range P2P. The true f is lower by the P2P share, which this file cannot give.

**What this file cannot show.** It has no gravity-versus-hydro cost split. Three
routes to one all fail. (1) Regression of step time on gas- and gravity-update
counts: the two counts correlate at 0.9956, condition number 1.4e7, and the fit
returns a negative cost per gravity update. (2) The long-range mesh flag (Props bit
256): all 644 mesh steps are also rebuild steps, so the mesh and tree-build costs
are inseparable here. (3) Steps with gravity active and gas inactive: 42 of 220,577.

So "gravity is the dominant component" rests on Schaller 2024 Fig. 20 and that
sentence in its §9.4, measured on their weak-scaling setup, not on this run. **To
measure it here: a task-debugging build, a few dozen steps, then
`tools/task_plots/analyse_tasks.py`.** Cheapest decisive measurement available, not
yet done.

Also from this file: 27.4% of the wall clock is in the 644 combined mesh-and-rebuild
steps — long-range gravity and tree construction, neither of which is the
short-range P2P work a GPU would take.

## What the SWIFT-GPU fork is building right now (checked 2026-07-28)

Their public repo is live development, not a paper snapshot. `gpu_master` last updated
2026-02-17 (a month after the RASTI paper). Active branches:

| Branch | Last commit |
|---|---|
| `selective_offloading` | **2026-07-17** |
| `fail_safe_for_large_cells_and_auto_GPU_mem_allocation` | 2026-07-03 |
| `merging_unique_sorts` | 2026-07-02 |
| `unique_sorting_and_bug_fixes_for_eagle_type_runs` | 2026-06-22 |

**`selective_offloading` is doing for hydro what we planned for gravity.**
`runner_GPU_offload_switch` (`src/runner_doiact_functions_hydro_gpu.h:334`) runs once per
step, recurses all local top-level cells counting active leaves, and returns
`n_active_leaves > pack_size * nr_threads` — offload only if there is at least one full
pack per thread, else run the whole step on CPU.

Their own TODOs admit it is provisional: *"Need to check that this is a good estimate"*
and *"Need to time this separately to ensure we are not wasting too much time here"*.
Also mid-bug as of 2026-07-17: *"Code hangs with stealing on"*.

**Coarser than pkdgrav3's**, and that matters for gravity: theirs is one all-or-nothing
decision per step on active-cell count; pkdgrav3 judges each interaction list on flop/byte.
With cell occupancy spanning 5 to 125,113 particles (Ivkovic's EAGLE25 measurement),
a per-step switch sends much thin work to the GPU alongside thick. Gravity likely needs
the finer criterion — but start with theirs, since it exists.

Also in that branch:
- **Cell deduplication when packing** — hash table (`hash_lookup_and_pack`, `my_index`,
  `n_unique`) so a cell in several task pairs is copied once. A *different* attack on the
  pack cost from "write the transfer buffer directly"; possibly complementary.
- `cuda_particle_kernels.cuh` **+1,374 lines** — substantial kernel rework since the paper.
- `tools/plot_gpu_timers/plotGPUtimers.py` — offload-cycle profiling tool, reusable as-is.
- `scheduler.c` +78, per-thread buffer sizing, GPU memory capacity checks.

**Accessor rework, partially done — relevant to how we write our first line of code.**
Ivkovic 2026 §3.2 proposes hiding every field access behind getters/setters so the memory
layout can be changed without touching physics. That layer IS in the fork for hydro:
`src/hydro/SPHENIX/hydro_part.h` has ~106 accessors (`part_get_x`, `part_set_v`, …) and
30 underscore-prefixed fields (`_x`, `_h`, `_v`), on both `gpu_master` and
`selective_offloading`. The struct is not yet split into substructs — accessors are the
prerequisite for that. **`gravity_part.h` has zero accessors and zero renamed fields.**

Two consequences: (1) if the layout rework pays off, someone must repeat it for `gpart`,
and that someone is us; (2) if we write gravity offload against `gpart` fields directly
while they are moving to accessors, we are on the wrong side of a convention they have
already committed to. **Ask them: do they plan the same treatment for `gpart`, and should
new code use accessors from the start?** Cheap for them to answer, changes our day one.

Still **no gravity code anywhere**. But they are ~3 months ahead on shared infrastructure
and changing it weekly. Contact them before starting; watch these branches.

## Historical note: GPU branches on upstream SWIFT

`github.com/SWIFTSIM/SWIFT` has two, both abandoned and never merged:
`GPU_swift` (last commit 2018-01-05) and `cuda_test` (2017-09-08). Early experiments,
dormant ~8 years, not a usable base. Worth knowing so nobody proposes reviving them.

## GPU kernel design comparison — the decisive section

### pkdgrav3 `cuda/cudapppc.cu` (Hopper-shaped)
- Block `(32, 8, 1)` = 256 threads. `threadIdx.x` = one **source** per warp lane;
  `threadIdx.y` = 8 **sinks**, loop strides by 8.
- Source tile staged to shared memory once via `cooperative_groups::memcpy_async`
  (hardware async copy, native on Hopper), reused across 8 sinks.
- `warpReduceAndStoreAtomicAdd<float,32>` — warp shuffle then one atomic/warp.
- AoSoA tile width 32 = warp width → fully coalesced (`gpu/workunit.h`).
- Host staging: page-aligned `std::aligned_alloc`, **not pinned** — correct for
  coherent NVLink-C2C (`gpu/hostdata.h`).
- 8 streams/device default, multi-device aware.
- `cscs-uenv/gh200/` Spack env with `cuda_arch=90` ships in-tree. **But all published
  measurements are on Piz Daint = Xeon E5-2690 v3 + Tesla P100 over PCIe**, including the
  2.1e9-particle run on 256 nodes (Meier 2026 §4). There are NO published pkdgrav3
  numbers on GH200. Sound guide to *structure*, not evidence about GH200 performance.
- **Single-source physics**: `gravity/pp.h` `EvalPP` etc. are
  `template<class F, class M>` with `PP_CUDA_BOTH` (`__host__ __device__`),
  compiled for CPU SIMD *and* CUDA *and* Metal. This is the key structural idea.
- CPU SIMD works on Grace via `core/sse2neon.h` (SSE intrinsics → NEON).

### SWIFT-GPU `src/cuda/gpu_launch.cu` (not yet tuned)
- `<<<num_blocks_x, 64>>>`, one thread per **sink**, serial j-loop from global
  memory. Disjoint j-ranges per thread → uncoalesced.
- No shared memory: `/* TODO: Do we want to allocate shared memory here? */` at
  all three launch sites. No async copy, no warp reduction.
- `cudaMallocHost` pinned buffers — right for PCIe, redundant on GH200.
- Physics duplicated: `device_functions.cuh` hardcodes cubic-spline SPHENIX
  constants, "TODO: This needs to become SPH flavour specific."

**Why cuGOTPM's threading cannot transfer (the load-bearing hardware fact).** A GPU runs
threads in lockstep groups of 32 (a warp); all 32 execute the same instruction at the same
moment. On a branch disagreement the hardware runs one side with the others disabled, then
swaps — serialised, half the machine idle. cuGOTPM gives each particle its own thread
walking the tree with its own open/reject decisions, so neighbours diverge within a few
steps and 32 lanes take 32 routes. pkdgrav3 puts all 32 lanes on one shared source list
differing only in *which entry* they read: same instruction, different data.

### cuGOTPM `gpu.Treewalk.mod2.cu` (wrong threading, cheapest arithmetic)
- One thread per particle, independent stackless walk → warps diverge as soon as
  two particles open different cells. Weakest threading design of the three.
- Whole tree re-uploaded per call; global timestep. Neither fits SWIFT.
- **Force law is 3 instructions**: `__log10f` (SFU) + 1 FMA + `tex1D` with
  hardware linear interpolation. No branch, no `rsqrt`, no polynomial.
- Tree flattened to `int2` sibling/daughter indices, negative = particle chain,
  read via `__ldg`. Pointer-free — the pattern if a device-side walk is ever wanted.
- `cuda_acquire_stream`: lock-free slot pool; **returns -1 when exhausted and the
  caller computes on the CPU instead**. Each slot owns its buffers. ~30 lines of
  heterogeneous co-execution with automatic load balance.

## Potter 2017 §3.3 — pkdgrav3's own CPU/GPU split rationale

Direct quote, and it is option C arrived at independently: *"we made the deliberate
decision to split the work between the CPU and GPU in a manner that compliments their
strengths. Walking a tree is geometrically complex, exhibits branch divergence, and
requires accessing tree nodes on remote processors. Conversely, evaluating interactions
and multipoles is ideal work for the GPU."*

**Runtime dispatch on arithmetic intensity**: *"PKDGRAV3 monitors the flop/byte ratio of
interaction lists as they are generated and in the rare case that this falls below an
optimal threshold then the work is instead issued directly to the CPU."* This is the
answer to the cell-occupancy spread (5 → 125,113 particles): do not send thin work to the
GPU at all. Finer-grained than the fork's per-step active-cell count.

Two implementation choices worth copying:
- **Mixed precision**: interactions evaluated in single, accumulated in double, giving
  ~10⁻⁵ RMS force error. Accuracy is protected at accumulation, which widens what is
  acceptable in the per-interaction arithmetic — directly relevant to the force-table
  question below.
- **Higher multipole order helps the GPU**: they use 4th-order moments (most tree codes
  use quadrupoles) and note that raising the order increases flop/byte, the FMA fraction,
  and work done per `rsqrt`. The more accurate method is the more GPU-friendly one.
  SWIFT exposes `--with-multipole-order`, so this is a lever we already have.

Other parameters: bucket size b ≈ 16, group size g = 64 CPU / **256 GPU** (Meier §2.3.2).
Opening criterion evaluated branchlessly with SIMD intrinsics, ~2% of runtime.

## Why pkdgrav3 has no pack bottleneck (structural, not an optimisation)

It never materialises data in a non-GPU layout. `ilBlockBase` is `scalar_t s[N]` with
N=32, SIMD-aligned; the tree walk calls `ilp.append(...)` and writes **each interaction
directly into the block layout the GPU reads**, as it walks. `ILP_PART_PER_BLK 32
/* Don't mess with this: see CUDA */`. A union presents the same memory to the CPU as
vector registers, so one buffer serves both. `copyBLKs` is then a bulk `memcpy` of
contiguous runs, not a gather.

SWIFT's gravity is halfway there — `gravity_cache` is already the flat per-attribute
layout, the hard part — but would be written once from the particle arrays and copied
*again* into a transfer buffer. **pkdgrav3 does one write where SWIFT would do two.**
Design the leaf-pair collection to write the transfer buffer directly, in GPU layout.
Cheaper to do now than to retrofit; attacks the 60–80% at its root.

Caveat: part of why they report no bottleneck is that their published runs are PCIe/P100,
where transfer dominates and packing is ~2% (Ivkovic Fig. 1). The structural argument
still holds on GH200.

## Meier 2026 reported performance (all on Piz Daint P100, not GH200)

Benchmark: Mars-sized body, 1M → 2e9 particles. Time per step follows O(N^4/3); dominated
by the three tree-walking operations (density, interface correction, force), ~1/3 each,
with gravity computed in the same pass as SPH forces.
- Strong scaling @ 2,048M particles: >80% efficiency to 512 nodes, 66% at 1,024
  (speedup 678, ~180k particles/thread).
- Strong scaling @ 256M: >80% only to 64 nodes, 56% at 256, 20% at 1,024; best 218 at 512.
- Weak scaling: >50% at 2M particles/node, >75% at 8M.

No particles/second figure. Nasar quotes theirs that way (~14.8M updates/s on one
GH200 vs 8.26M CPU baseline, 1.8x) on a gravity-free Gresho–Chan vortex, while
Meier's benchmark is a self-gravitating Mars-sized body with gravity in the same pass
as the SPH forces. Neither the units nor the physics content match, so the two papers
cannot be compared.

## The force-table question (revised 2026-07-27 — earlier version was wrong)

SWIFT's `runner_iact_grav_pp_truncated` costs `rsqrt` + branch on `r2>=h2` +
2 softening polynomials + `expf(-u2)` + Abramowitz–Stegun 7.1.26 erfc rational
approximation, **per pair**. cuGOTPM does the whole radial law in 3 instructions.

Initially assessed as the highest-leverage change. That was wrong on two counts:

1. **Amdahl.** If pack/unpack is 80% of the cycle, a free kernel buys 1.25x.
2. **The texture path has aged badly.** H100 SM: 128 FP32 lanes, ~1/4 that in
   SFU, **4 texture units**. Kepler (cuGOTPM's target) was ~12:1 arithmetic:
   texture; Hopper is ~32:1. One `tex1D` per pair can go texture-bound *below*
   the rate the arithmetic version sustains.

But the 80% is an SPH number and does not carry over. SWIFT splits cells above
400 particles (`space_splitsize_default`, `src/space.h:49`), so a gravity leaf
pair at ~200/side is ~40,000 interactions against ~10 KB → **~125 flop/byte**,
where GH200's 900 GB/s against H100 FP32 balances nearer **66**. Gravity P2P
should be compute-bound, ~2x past balance. Needs confirming against real
leaf-occupancy distributions.

**What survives**: take pkdgrav3's *branchless* masked softening (removes warp
divergence, stays on the FP32 pipe, no accuracy argument, no table). The table
becomes a measurement question confined to the `expf`+erfc truncation term, and
the version to test is `__ldg` + FMA interpolation in FP32 (load path), not
`tex1D` (4 TMUs). Texture filtering quantises the interpolation weight to ~9
bits — too coarse for a code validated against direct summation. Gate on
`--enable-gravity-force-checks`.

## Why option A (transplant pkdgrav3 gravity) is rejected

`cudaInteract` takes a tile from an **interaction list**, built by
`gravity/walk2.cxx` during pkdgrav3's own tree traversal, which needs its tree,
which is built over ORB subdomains, with remote cells fetched through the MDL
cache. SWIFT has none of these — it has cell-pair tasks from a scheduler and
makes accept/open decisions inline rather than recording them.

So 1,400 lines of CUDA becomes ~11,100 (`gravity/` 8,795 + `cuda/` 1,367 +
`gpu/` 925), displacing SWIFT's 10,444-line FMM (invoked in 21 places in the
recursive walk) plus the mesh, MPI proxies and timestep. Three further blocks:
data structures disagree (`gravity/moments.c` 2,922 vs `src/multipole.h` 3,168,
different conventions); pkdgrav3's rungs cannot express SWIFT's per-particle
`active` flag — **wrong, corrected 2026-07-28**: SWIFT (`timebin_t`, 56 bins,
`1LL << (bin+1)`) and pkdgrav3 (`uRung`, 2^-l) use the *same* per-particle power-of-two
scheme. Good news: pkdgrav3's GPU design is already proven under SWIFT's activity
pattern. cuGOTPM is the outlier with its global timestep. 18–30 person-months and the
result is not SWIFT.

## Why option C is feasible

1. **Clean seam.** `runner_dopair_recursive_grav` does MAC decisions on CPU;
   direct P2P only fires at leaf pairs (`runner_dopair_grav_pp`). Keep recursion,
   M2L, M2P, `grav_down` on CPU.
2. **`gravity_cache` already packs.** SoA x, y, z, epsilon, m in / a_x, a_y, a_z,
   pot out, VEC_SIZE-padded, with `active` and `use_mpole` flags
   (`gravity_cache.h`). The hydro port had to invent its packing.
3. **Format fits gravity better than hydro.** All-pairs between adjacent leaves:
   regular ranges, no sorted-list pruning, no variable smoothing length.
4. **Validation.** `--enable-gravity-force-checks` compares against direct summation.

## Sizing (option C)

| Component | Lines | Precedent |
|---|---|---|
| `__host__ __device__` interaction header (full + truncated + kernels) | 400–500 | SWIFT `gravity_iact.h` 204; pkdgrav3 `gravity/pp.h` 456 |
| `gpu_part_send/recv_grav` structs + pack/unpack | 600–800 | fork `gpu_part_structs.h` 143 + `gpu_part_pack_functions.h` 305 |
| CUDA kernel + launch (warp-per-source, shared-mem tile, warp reduce) | 400–500 | pkdgrav3 `cudaInteract` + `cuda/reduce.h` 114; fork `gpu_launch.cu` 194 |
| `task_subtype_gpu_grav_pp` + dependency edges into `grav_down`/`end_force` | 400–600 | fork added 123 lines for 3 hydro subtypes |
| Leaf-pair collection variant of `runner_dopair_recursive_grav` | 300–400 | `runner_doiact_grav.c:2238–2416`; reuse `gpu_pack_metadata` ci/cj_leaves |
| Tabulated force splines (only if measurement justifies) | 200–250 | cuGOTPM `build_force_textures` + `PARTICLEFORCE` (~120) |
| Slot pool with CPU fallback | 100–150 | cuGOTPM `cuda_acquire_stream` (~30) |
| Build system, runtime flag, CPU fallback | 150–200 | fork `--with-cuda` |
| Wire into exact-force checks | 100–150 | already exists CPU-side |
| **Total, rounded up** | **3,000–5,000** | only a modest fraction on the GPU itself |

Effort: ramp-up 3–4wk (2–3 assisted), kernel+pack 3–4wk (1.5–2), task machinery
5–8wk (4–6), validation 4–6wk (3–5), GH200 tuning 4–6wk (3–5), rebase overhead
~15% continuous. **Prototype 5–7 months alone / 4–5 assisted; production 9–12 / 7–9.**

## Phase order

0. **Measure the P2P share of a step on our own problem** (added 2026-07-30).
   Task-debugging build, a few dozen steps of the 25 Mpc/h setup at several points in
   the step hierarchy, then `tools/task_plots/analyse_tasks.py`. Days, no GPU, and it
   sets the ceiling for everything after it. If short-range P2P is half or more of a
   large step, offload it alone; if it is a quarter, M2L/M2P have to go too or the
   whole split does not repay its maintenance.
1. Build the fork on GH200, reproduce published hydro numbers. **Contact Nasar,
   Ivkovic, Schaller about their gravity roadmap** — `EAGLE_w_gravity` suggests
   it is planned. Blocking.
2. Single-source the interaction kernel (`__host__ __device__` returning a result
   struct, pkdgrav3's `EvalPP` pattern). Verify CPU path bit-identical. Highest
   structural value; worth doing even if the rest stalls.
3. Offload path with a naive one-thread-per-sink kernel. Structs, leaf-pair
   collection, `task_subtype_gpu_grav_pp`, CPU fallback behind a runtime flag.
   Dependency wiring is the real work. **Write the transfer buffer directly in GPU
   layout during leaf collection — do not add a separate pack step and plan to remove
   it later** (moved up from phase 6, 2026-07-30). On Nasar's own measurement the pack
   step is the difference between 1.8x and ~5.2x, and it is cheaper to avoid than to
   retrofit. Also decide per step whether to offload at all, with the CPU fallback, in
   this first version rather than in phase 6: 42.8% of our wall clock is in steps too
   small to offload.
4. **Validate before optimising.** Exact-force → energy conservation → P(k) vs CPU.
5. Hopper-shaped kernel (warp-per-source, shared-mem tile, warp reduce), then
   pkdgrav3's branchless softening. **Then profile** before considering any table.
6. Refine the offload decision from the crude per-step version in phase 3. Slot pool
   with CPU fallback (cuGOTPM pattern: GPU *availability*), plus a worth-offloading
   test (pkdgrav3 pattern: work *suitability*, per interaction list). Start from the
   fork's `runner_GPU_offload_switch` since it exists, then refine — their per-step
   active-cell count is too coarse for gravity's occupancy spread. Their hash-table
   cell deduplication is the remaining staging reduction once the buffer is already
   written directly. Retest reading `gravity_cache` in place over C2C against
   Ivkovic's negative result. Measure the effect of batch size on the dependency
   releases: Nasar's time outside task execution rose 129 → 203 ms from batching
   alone.
7. Optional: M2P on device; cuGOTPM's flattened `int2` tree if a device walk is
   ever wanted.

## Grace-side finding (separate, unbudgeted)

**SWIFT has no ARM SIMD.** `src/vector.h` branches only on `HAVE_AVX512F`,
`HAVE_AVX`, `HAVE_SSE2`, else `#define VEC_SIZE 4` scalar. Nothing in `src/`
mentions NEON or SVE outside `cycle.h` and `memswap.h`. Gravity P2P survives
(plain C over aligned SoA with `swift_assume_size`, auto-vectorises to NEON) but
every path using `vector.h` intrinsics runs scalar on Grace. On GH200 the CPU
does the tree walk and packing, so this is not a side issue. Adding an SVE branch
benefits the whole code. pkdgrav3 has this covered via `core/sse2neon.h`.

## Communication status (2026-07-28)

- Artifact sent to BK answering his original question. Made public by JG so collaborators
  can open it.
- Da replied asking whether the fork's public branches are current and whether to request
  a newer version. Answered: public repo *is* the live development (branches updated this
  month); no hidden version to request. Also pointed to `swiftgpupacksim` as a separate
  pack/unpack experiment detached from the SWIFT codebase.
- JG's position, agreed: gravity is almost certainly on their roadmap, and **some
  duplicated effort is unavoidable** if we build our own. Contact them for the roadmap,
  not to avoid duplication.
- Ho Seong, Gain and Jie looped in; Zoom call being scheduled via when2meet. Da to
  summarise MPI-enabled monofonIC + particle-merging zoom-in ICs and production cost
  estimates.
- **Not yet asked of Manchester**: the `gpart` accessor question above, and whether they have
  tried writing the transfer buffer directly during task collection rather than packing
  afterwards. Their newest branch attacks the pack cost by deduplicating repeated cells
  instead, which suggests they have not.
- 2026-07-30: artifact republished at the same URL with the corrected performance numbers
  (gravity-free benchmark, 3.5-vs-7.5 meaning, our own step-size ceiling) and a new
  "What speedup to expect" section. BK's existing link resolves to the corrected version.
  BK's reply — agreeing on the CPU base, and warning that SWIFT's gain comes from task
  delegation so the GPU inherits the same task shapes — is the point this correction
  confirms. Reply to BK not yet sent.

## Risks

- SWIFT-GPU is a moving prototype (`cuda_config.h` calls itself temporary);
  upstream moves fast. Their Oct 2025 sync suggests rebasing is workable.
- **Coordination, not collision.** Gravity is unclaimed, but the fork's shared
  infrastructure (selective offloading, pack path, kernels) is changing weekly as of
  July 2026. Contact them early and track `selective_offloading` and the sorting branches.
- **License is NOT a constraint** (settled 2026-07-28). Use is private, no release
  planned. GPLv3 §2 permits private modification and use "without conditions"; the
  obligations attach to *conveying*. cuGOTPM has no licence file (= default copyright,
  no rights granted) but **Juhan has agreed to its use**. Caveats: "private" means within
  the legal entity — sharing with another institution is conveying; keep file headers and
  track provenance, cheap insurance if plans change. Still reimplement rather than copy,
  for the engineering reason above, not a legal one.
- cuGOTPM repo is untidy: ~6,800 lines of `.cu` across near-duplicate variants
  (`gpu.Treewalk.cu`, `.mod1`, `.mod2`, `GPU/...mod3`, `.mod31`, `.mod32`).
  Fine to read, poor to depend on.
- Table accuracy unresolved; must be gated on force checks, not assumed.
