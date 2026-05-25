# Evaluation Scripts

This folder contains unified evaluation entry points for thesis experiments.

The goal is to keep one runner for diffusion and baseline metrics while
reusing older root-level metric implementations for compatibility.

- `run_eval_metrics.sh` is the single metrics runner for:
  - fairness profile (`--profile fairness`, topology on by default)
  - dt-family profile (`--profile dt_family`)
  - baseline profile (`--profile baseline --baseline <pix2pix|cyclegan|vqgan>`)
- `metric_crack_topology.py` adds crack-specific topology metrics for connectivity, endpoints, branchpoints, skeleton distance, and width distribution.
- `eval_select_ids.py` standardizes eval id sampling from DT maps.

The older metric implementations remain at the repository root for compatibility:

- `/work/metric_fid_psnr.py`
- `/work/metric_struct_align.py`
