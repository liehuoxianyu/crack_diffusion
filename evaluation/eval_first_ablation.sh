#!/usr/bin/env bash
set -euo pipefail

# Generate images for the first thesis ablation runs:
#   dt_updated_prompt, topology, appearance, topology_weighted

OUTDIR="${OUTDIR:-/work/outputs/diffusion_eval_first_ablation}"
CKPTS="${CKPTS:-500 1000 1500 2000}"
MODES="${MODES:-dt_updated_prompt topology appearance topology_weighted}"

python /work/eval_diffusion_refresh/eval_dt_family.py \
  --outdir "$OUTDIR" \
  --ckpts $CKPTS \
  --modes $MODES
