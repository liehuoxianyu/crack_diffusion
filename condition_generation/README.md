# Condition Generation

This folder is the new home for condition-map generation.

- `generate_topology_conditions.py` creates structure-focused conditions from binary crack masks.
- `generate_appearance_conditions.py` creates appearance-focused conditions from pavement images.
- `generate_synthetic_masks.py` creates synthetic binary crack masks for segmentation dataset expansion.
- `old/` keeps older condition-generation scripts for traceability.

Recommended first experiments:

```bash
python /work/condition_generation/generate_topology_conditions.py \
  --mask-dir /CrackTree260/cond_mask \
  --out-dir /CrackTree260/cond_topology

python /work/condition_generation/generate_appearance_conditions.py \
  --image-jsonl /CrackTree260/train_linux.jsonl \
  --out-dir /CrackTree260/cond_appearance
```

Synthetic mask expansion:

```bash
python /work/condition_generation/generate_synthetic_masks.py \
  --real-mask-dir /CrackTree260/cond_mask \
  --out-dir /work/outputs/synthetic_crack_dataset/masks \
  --num-masks 500 \
  --size 512 \
  --seed 42 \
  --write-default-conditions
```

This writes binary labels under `masks/`, provenance under `metadata.jsonl`, a real-vs-synthetic
statistics report, a mask contact sheet, and ControlNet-ready `cond_dt/` and `cond_topology/`
siblings when `--write-default-conditions` is set.
