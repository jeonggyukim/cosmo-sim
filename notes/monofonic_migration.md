# monofonIC migration plan

## Current state (2026-04)

The IC generator at `~/Dropbox/Projects/MUSIC2` — despite the directory
name — is the **legacy multi-scale MUSIC** (Hahn & Abel 2011), upstream
repo `cosmo-sims/MUSIC`. The "v2" in its README is the 2.x release line
of the same code, not a different algorithm.

Config keys in `conf/CV_22_MUSIC_template.conf` confirm this:
`levelmin`, `levelmin_TF`, `use_2LPT`, `ZeroRadiation`, `[setup]`,
`[random]` with `seed[N]=...` are all legacy-MUSIC keys.

### What this pipeline does NOT include

- **PLT (Particle Linear Theory) correction** — Joyce & Marcos 2007;
  Garrison et al. 2016. A lattice-preloaded particle system has
  anisotropic eigenmodes that differ from continuum growing modes near
  k_Ny by O(10–20%). Legacy MUSIC applies standard ZA/2LPT amplitudes,
  so mode amplitudes near k_Ny are biased and the bias stays baked into
  the IC as the simulation evolves.
  - grep for `PLT`, `particle_linear`, `Joyce`, `Marcos`, `Garrison` in
    `src/*.cc`, `src/*.hh`: no hits. The only match is a commented-out
    `HDFWriteGroupAttribute(..., "PLT", CMAKE_PLT_STR)` in
    `src/plugins/output_swift.cc:811`, inert.
- **3LPT** — legacy MUSIC is 1LPT/2LPT only.
- **Paired-and-fixed ICs** (Angulo & Pontzen 2016) as a first-class
  option. The template has `fix_mode_amplitude = yes` which does amplitude
  fixing, but not seed-paired inversion.

### Impact on current measurements

At z_start=200 the displacements are tiny (σ_Ψ ≪ d), so particles sit
almost exactly on the lattice and the PLT bias is partly hidden: the
measured mode amplitudes near k_Ny differ from the continuum prediction
by the eigenvalue ratio, but the deviation is masked by the Poisson
floor. It becomes visible once the simulation evolves to lower z.

In the P(k) ratio panel the signature is upward curvature in
P_meas / P_theory as k → k_Ny. Cap trust at ~0.5·k_Ny until PLT is
added.

## monofonIC target

Repo: `https://github.com/cosmo-sims/monofonIC`. Intended as the
unigrid successor to MUSIC. Drop-in for non-zoom runs, not for nested
zoom ICs (MUSIC is still the tool for those).

Key differences from legacy MUSIC:

| Feature | legacy MUSIC | monofonIC |
|---|---|---|
| LPT order | 1, 2 | 1, 2, 3 |
| PLT correction | no | yes (on by default) |
| Back-scaling | yes (ZeroRadiation) | yes (and forward option) |
| Zoom / nested | yes | no (unigrid only) |
| Config format | INI, `[setup]`/`[cosmology]`/`[random]` | INI, different keys |
| Output plugins | Gadget, AREPO, Swift, RAMSES, ART, ENZO, ... | Gadget, AREPO, Swift, HACC, generic |
| Transfer fns | CAMB, CLASS, E&H, ... | CLASS (built-in), file |

Key monofonIC config keys (for future reference):

```
[setup]
GridRes         = 256         # N per side
BoxLength       = 256         # Mpc/h
zstart          = 200
LPTorder        = 3
DoBaryonVrel    = no
DoFixing        = yes         # Angulo & Pontzen fixing
DoInversion     = no          # seed pair partner
ParticleLoad    = sc          # sc / bcc / glass
SymplecticPT    = no

[cosmology]
ParameterSet    = Planck2018EE+BAO+SN
...

[random]
generator       = NGENIC      # or PANPHASIA, MUSIC1
seed            = 12345
```

## Migration plan

1. **Add monofonIC as an alternative to legacy MUSIC** (both kept; MUSIC
   needed for any future zoom work).
   - New build script `build-monofonic.sh` paralleling `build-music.sh`.
     Source directory resolution order: `$MONOFONIC_SOURCE_DIR`,
     `../monofonIC`, else clone from GitHub.
   - New template `conf/CV_22_monofonIC_template.conf` with the keys
     above.
   - New `scripts/make_monofonic_conf.py` paralleling
     `make_music_conf.py`.
2. **Extend `run_pipeline.sh`** with a `--ic-code {music,monofonic}`
   flag (default `music` for now, flip to `monofonic` once validated).
   Steps 1 and 3–4 branch on the choice; the rest of the pipeline is
   unchanged (CLASS, ξ, ψ, P(k), plotting all operate on the SWIFT
   HDF5 output, which both codes write).
3. **Validate PLT effect** by running identical cosmology + seed +
   box with both codes and overplotting P_meas / P_theory vs. k / k_Ny.
   The monofonIC curve should stay flatter near k_Ny.
4. **Document** the PLT caveat and the two-code setup in `CLAUDE.md`
   and `cosmo_ic.tex` §11 (which already covers MUSIC2/monofonIC
   algorithms; extend with a subsection on the concrete pipeline
   plumbing).

## Other artifacts worth checking while migrating

From the earlier audit:

- **LPT transient** from 1LPT → use 2LPT or 3LPT (monofonIC default 3).
- **Species-dependent transfer**: confirm whether sourcing from total
  matter Tcb or CDM-only in monofonIC, match CLASS output accordingly.
- **Primordial / N_ur match** between IC generator and CLASS — already
  auto-matched via `input_class_parameters.ini`, re-verify with
  monofonIC's CLASS emission.
- **CIC deconvolution × aliasing residual** near k_Ny in `compute_pk.py`
  — intrinsic to the estimator, unchanged by the IC code switch.
- **Low-k sample variance** — mitigate with multi-seed averaging once
  monofonIC paired-and-fixed runs are wired in.
