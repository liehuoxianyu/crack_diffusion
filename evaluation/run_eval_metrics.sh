#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Unified metrics runner for diffusion and baseline evaluations.

Usage:
  bash /work/evaluation/run_eval_metrics.sh [options]

Options:
  --profile <fairness|dt_family|baseline>  Preset defaults (default: fairness)
  --baseline <pix2pix|cyclegan|vqgan>      Required when profile=baseline
  --eval-root <path>                       Override eval output root
  --steps <csv>                            Override step labels, e.g. 500,1000
  --modes <csv>                            Override modes, e.g. dt,dt_tag
  --with-topology                          Force topology metric on
  --without-topology                       Force topology metric off
  --eval-ids-txt <path>                    Override eval ids txt path
  --real-image-jsonl <path>                Override real image jsonl path
  --cond-root-binary <path>                Override binary condition directory
  --cond-root-dt <path>                    Override dt condition directory
  -h, --help                               Show this help
EOF
}

PROFILE="fairness"
BASELINE=""
OVERRIDE_EVAL_ROOT=""
OVERRIDE_STEPS=""
OVERRIDE_MODES=""
TOPOLOGY_OVERRIDE=""

EVAL_IDS_TXT="${EVAL_IDS_TXT:-/CrackTree260/eval_ids.txt}"
REAL_IMAGE_JSONL="${REAL_IMAGE_JSONL:-/CrackTree260/train_linux.jsonl}"
COND_ROOT_BINARY="${COND_ROOT_BINARY:-/CrackTree260/cond_mask}"
COND_ROOT_DT="${COND_ROOT_DT:-/CrackTree260/cond_dt}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --baseline)
      BASELINE="${2:-}"
      shift 2
      ;;
    --eval-root)
      OVERRIDE_EVAL_ROOT="${2:-}"
      shift 2
      ;;
    --steps)
      OVERRIDE_STEPS="${2:-}"
      shift 2
      ;;
    --modes)
      OVERRIDE_MODES="${2:-}"
      shift 2
      ;;
    --with-topology)
      TOPOLOGY_OVERRIDE="1"
      shift
      ;;
    --without-topology)
      TOPOLOGY_OVERRIDE="0"
      shift
      ;;
    --eval-ids-txt)
      EVAL_IDS_TXT="${2:-}"
      shift 2
      ;;
    --real-image-jsonl)
      REAL_IMAGE_JSONL="${2:-}"
      shift 2
      ;;
    --cond-root-binary)
      COND_ROOT_BINARY="${2:-}"
      shift 2
      ;;
    --cond-root-dt)
      COND_ROOT_DT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$PROFILE" in
  fairness)
    EVAL_ROOT="/work/outputs/diffusion_eval_fairness"
    STEPS="500,1000,1500,2000"
    MODES="dt,dt_cafe,dt_tag,dt_cafe_tag"
    RUN_TOPOLOGY="1"
    ;;
  dt_family)
    EVAL_ROOT="/work/outputs/diffusion_eval_dt_family"
    STEPS="1000,2000"
    MODES="dt,dt_tag,dt_cafe_tag"
    RUN_TOPOLOGY="0"
    ;;
  baseline)
    BASELINE="${BASELINE:?--baseline is required when --profile baseline}"
    EVAL_ROOT="/work/outputs/GAN/eval/${BASELINE}"
    MODES="binary"
    RUN_TOPOLOGY="0"
    case "$BASELINE" in
      pix2pix) STEPS="500,1000,1500,2000" ;;
      cyclegan) STEPS="25,50,75,100,125,150,175,200" ;;
      vqgan) STEPS="5000,10000,15000,20000" ;;
      *)
        echo "Unsupported baseline: $BASELINE" >&2
        echo "Allowed values: pix2pix, cyclegan, vqgan" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported profile: $PROFILE" >&2
    echo "Allowed values: fairness, dt_family, baseline" >&2
    exit 1
    ;;
esac

if [[ -n "$OVERRIDE_EVAL_ROOT" ]]; then
  EVAL_ROOT="$OVERRIDE_EVAL_ROOT"
fi
if [[ -n "$OVERRIDE_STEPS" ]]; then
  STEPS="$OVERRIDE_STEPS"
fi
if [[ -n "$OVERRIDE_MODES" ]]; then
  MODES="$OVERRIDE_MODES"
fi
if [[ -n "$TOPOLOGY_OVERRIDE" ]]; then
  RUN_TOPOLOGY="$TOPOLOGY_OVERRIDE"
fi

if [[ ! -d "$EVAL_ROOT" ]]; then
  echo "Missing eval root: $EVAL_ROOT" >&2
  exit 1
fi

export EVAL_ROOT
export EVAL_IDS_TXT
export REAL_IMAGE_JSONL
export COND_ROOT_BINARY
export COND_ROOT_DT
export STEPS
export MODES

python /work/metric_fid_psnr.py
python /work/metric_struct_align.py

if [[ "$RUN_TOPOLOGY" == "1" ]]; then
  python /work/evaluation/metric_crack_topology.py \
    --eval-root "$EVAL_ROOT" \
    --eval-ids "$EVAL_IDS_TXT" \
    --mask-dir "$COND_ROOT_BINARY" \
    --modes "$MODES" \
    --steps "$STEPS"
fi
