# First Ablation Summary

## Scope

Runs:

- `dt_updated_prompt`
- `topology`
- `appearance`
- `topology_weighted`

All runs use `/CrackTree260/train_linux_updated.jsonl`, SD1.5-ControlNet, batch size 2, gradient accumulation 4, and checkpoints at 500/1000/1500/2000 steps.

Outputs:

- Generated images: `/work/outputs/diffusion_eval_first_ablation`
- FID/PSNR/SSIM/LPIPS: `/work/outputs/diffusion_eval_first_ablation/metric_fid_psnr_summary.csv`
- Existing structural metrics: `/work/outputs/diffusion_eval_first_ablation/metrics_summary_by_step.csv`
- Crack topology metrics: `/work/outputs/diffusion_eval_first_ablation/metric_crack_topology_summary.csv`

## Main Observations

- `appearance` is strongest on generic visual quality. Its best FID is `156.26` at step 2000, and its best LPIPS is `0.1881` at step 2000.
- `dt_updated_prompt` is strongest on the existing structural Chamfer metric. Its best Chamfer is `9.34` at step 500.
- `topology` and `topology_weighted` do not outperform `dt_updated_prompt` on the current structure metrics. This suggests the current topology RGB encoding is not yet better than the smoother DT condition.
- The weighted loss did not clearly improve the topology run under the current metric setup. Its best FID is `174.47` at step 1000, but structure metrics remain close to or worse than the non-weighted topology run.

## Best Checkpoints By Metric

- Best FID: `appearance` step 2000, `156.26`.
- Best LPIPS: `appearance` step 2000, `0.1881`.
- Best PSNR: `appearance` step 500, `14.98`.
- Best existing structural Chamfer: `dt_updated_prompt` step 500, `9.34`.
- Best crack-topology skeleton Chamfer: `dt_updated_prompt` step 500, `12.26`.
- Best endpoint F1: `dt_updated_prompt` step 500, `0.0135`.
- Best branchpoint F1: `dt_updated_prompt` step 500, `0.0229`.

## Interpretation

The first ablation supports a split conclusion:

- Appearance-derived conditions help image realism and perceptual quality.
- DT remains a stronger structure-control condition than the current topology encoding.
- The topology idea should not be abandoned, but the current `skeleton + DT heat + width` RGB condition is not enough to claim improvement.

For the thesis, the safer next step is to present `appearance` as the successful new condition branch, then refine topology with either a different channel layout or a combined condition.

## Recommended Next Steps

1. Inspect generated samples visually, especially `appearance@2000`, `dt_updated_prompt@500`, `topology@500`, and `topology_weighted@500`.
2. Try a combined RGB condition: `DT heat + skeleton + appearance shading`, rather than topology-only or appearance-only.
3. Reconsider the topology metric extraction method, because Canny-derived endpoints/branchpoints are very noisy and component counts are extremely large.
4. Keep CAFE/TAG as later optional ablations, not the next immediate priority.
