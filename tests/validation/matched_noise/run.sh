#!/usr/bin/env bash
# Hahn 2011 §4.3.1 matched-noise validation of MUSIC2-anisotropic-zoom.
# Key knobs: kaveraging=no + density_boundary=yes + kspace_TF=no.
# Both zoom and unigrid share seed[levelmin]/seed[levelmax] integers; with
# kaveraging=no the per-cell noise is coord-deterministic, and
# density_boundary=yes adds the three-term δ assembly that brings the patch
# interior δ(q) residual to ~10^-4 σ.
#
# Default: 1LPT.  Use --use-2lpt for the 2LPT variant.
set -euo pipefail

SEED_COARSE=12345
SEED_FINE=23456
BOXLENGTH=100
ZSTART=45
LEVELMIN=8
LEVELMAX=9
REF_EXTENT=0.2
USE_2LPT=yes
PADDING=16
ACCURACY=1e-9
SMOOTH=5
RUN_DIR="$HOME/Documents/music_validation"
TAG=matched_noise
OUT=""   # auto-derived from parameters if empty: see below
MUSIC_BIN="$HOME/Dropbox/Projects/MUSIC2-anisotropic-zoom/build/MUSIC"
SKIP_MUSIC=0
WITH_SWIFT=0   # also emit SWIFT-format particles (Eulerian diagnostics); off by default

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed-coarse) SEED_COARSE=$2; shift 2;;
    --seed-fine)   SEED_FINE=$2;   shift 2;;
    --boxlength)   BOXLENGTH=$2;   shift 2;;
    --zstart)      ZSTART=$2;      shift 2;;
    --levelmin)    LEVELMIN=$2;    shift 2;;
    --levelmax)    LEVELMAX=$2;    shift 2;;
    --ref-extent)  REF_EXTENT=$2;  shift 2;;
    --use-2lpt)    USE_2LPT=yes;   shift 1;;
    --use-1lpt)    USE_2LPT=no;    shift 1;;
    --padding)     PADDING=$2;     shift 2;;
    --accuracy)    ACCURACY=$2;    shift 2;;
    --smooth)      SMOOTH=$2;      shift 2;;
    --run-dir)     RUN_DIR=$2;     shift 2;;
    --out)         OUT=$2;         shift 2;;
    --music)       MUSIC_BIN=$2;   shift 2;;
    --skip-music)  SKIP_MUSIC=1;   shift 1;;
    --with-swift)  WITH_SWIFT=1;   shift 1;;
    -h|--help)     sed -n '2,11p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

LPT_TAG=$([[ "$USE_2LPT" == "yes" ]] && echo 2lpt || echo 1lpt)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$MUSIC_BIN" ]] || { echo "MUSIC binary not found at $MUSIC_BIN" >&2; exit 1; }

if [[ -z "$OUT" ]]; then
  OUT="$HOME/Documents/music_validation/figures/matched_noise_${LPT_TAG}_l${LEVELMIN}-${LEVELMAX}_re${REF_EXTENT}_b${BOXLENGTH}_z${ZSTART}_s${SEED_COARSE}-${SEED_FINE}.png"
fi

render() {
  local tpl=$1 fmt=$2 fname=$3 out=$4
  sed \
    -e "s/{BOXLENGTH}/$BOXLENGTH/g" \
    -e "s/{ZSTART}/$ZSTART/g" \
    -e "s/{LEVELMIN}/$LEVELMIN/g" \
    -e "s/{LEVELMAX}/$LEVELMAX/g" \
    -e "s/{REF_EXTENT}/$REF_EXTENT/g" \
    -e "s/{SEED_COARSE}/$SEED_COARSE/g" \
    -e "s/{SEED_FINE}/$SEED_FINE/g" \
    -e "s/{USE_2LPT}/$USE_2LPT/g" \
    -e "s/{PADDING}/$PADDING/g" \
    -e "s/{ACCURACY}/$ACCURACY/g" \
    -e "s/{SMOOTH}/$SMOOTH/g" \
    -e "s/{FORMAT}/$fmt/g" \
    -e "s/{FILENAME}/$fname/g" \
    "$tpl" > "$out"
}

mkdir -p "$RUN_DIR"
for d in "${TAG}_zoom_${LPT_TAG}" "${TAG}_unigrid_${LPT_TAG}"; do
  mkdir -p "$RUN_DIR/$d"
done

# For each (kind = zoom | unigrid), invoke MUSIC2 in format=generic to produce
# the Lagrangian δ(q), Ψ_x,y,z grids that plot_residuals.py reads.  With
# --with-swift, also invoke MUSIC2 in format=swift (writes Eulerian SWIFT
# particles) in the same dir; the second run reuses the wnoise_NNNN.bin from
# the first so the noise realisation is identical across formats.  The swift
# output is not used by plot_residuals.py — only enable it if you need
# particle-form data for an Eulerian diagnostic.
run_pair() {
  local kind=$1 tpl=$2
  local dir="$RUN_DIR/${TAG}_${kind}_${LPT_TAG}"
  local stem="${TAG}_${kind}_${LPT_TAG}"
  render "$tpl" generic "${stem}_g.hdf5" "$dir/run_g.conf"
  if [[ "$SKIP_MUSIC" -eq 1 && -f "$dir/${stem}_g.hdf5" ]]; then
    echo "[skip-music] $kind generic"
  else
    echo "=== $kind generic ==="
    ( cd "$dir" && "$MUSIC_BIN" run_g.conf > music_generic.log 2>&1 )
  fi
  if [[ "$WITH_SWIFT" -eq 1 ]]; then
    render "$tpl" swift "${stem}.hdf5" "$dir/run.conf"
    if [[ "$SKIP_MUSIC" -eq 1 && -f "$dir/${stem}.hdf5" ]]; then
      echo "[skip-music] $kind swift"
    else
      echo "=== $kind swift ==="
      ( cd "$dir" && "$MUSIC_BIN" run.conf > music_swift.log 2>&1 )
    fi
  fi
}

run_pair zoom    "$SCRIPT_DIR/zoom.conf.template"
run_pair unigrid "$SCRIPT_DIR/unigrid.conf.template"

mkdir -p "$(dirname "$OUT")"
# plot_residuals.py reads RUN_DIR=~/Documents/music_validation directly;
# everything run-specific (geometry, paths, margin) comes through env vars.
# Geometry (PL_xyz, PS_xyz) is parsed from MUSIC2's music_generic.log inside
# the script, so non-cubic patches and other LEVELMAX values work too.
OUT="$OUT" TAG="$TAG" LPT_TAG="$LPT_TAG" \
  LEVELMIN="$LEVELMIN" LEVELMAX="$LEVELMAX" REF_EXTENT="$REF_EXTENT" \
  conda run -n cosmo python "$SCRIPT_DIR/plot_residuals.py"
