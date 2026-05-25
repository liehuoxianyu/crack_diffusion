# First Ablation Step-2000 Comparison

This comparison fixes all methods at `checkpoint-2000`.

## Generic Image Metrics

| mode | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
| --- | ---: | ---: | ---: | ---: |
| `appearance` | 13.67 | 0.0562 | 0.1881 | 156.26 |
| `dt_updated_prompt` | 11.93 | 0.0363 | 0.2370 | 208.23 |
| `topology` | 11.63 | 0.0365 | 0.2290 | 211.72 |
| `topology_weighted` | 12.04 | 0.0391 | 0.2189 | 193.43 |

Step-2000 takeaway: `appearance` is clearly best for visual/perceptual quality at the fixed final checkpoint.

## Existing Structural Metrics

| mode | Chamfer ↓ | Recall ↑ | IoU_r ↑ | Dice_r ↑ |
| --- | ---: | ---: | ---: | ---: |
| `appearance` | 12.76 | 1.0000 | 0.0452 | 0.0825 |
| `dt_updated_prompt` | 9.75 | 1.0000 | 0.0452 | 0.0825 |
| `topology` | 12.79 | 1.0000 | 0.0452 | 0.0824 |
| `topology_weighted` | 12.79 | 1.0000 | 0.0452 | 0.0825 |

Step-2000 takeaway: `dt_updated_prompt` remains strongest on the existing Chamfer structure metric. IoU/Dice are almost tied across methods.

## Crack-Topology Metrics

| mode | Skeleton Chamfer ↓ | Component Error ↓ | Endpoint F1 ↑ | Branch F1 ↑ | Width MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `appearance` | 12.88 | 10012.05 | 0.0083 | 0.0118 | 0.1975 |
| `dt_updated_prompt` | 12.95 | 9028.60 | 0.0071 | 0.0118 | 0.3801 |
| `topology` | 12.91 | 9388.50 | 0.0075 | 0.0118 | 0.3069 |
| `topology_weighted` | 12.90 | 10232.60 | 0.0076 | 0.0120 | 0.3291 |

Step-2000 takeaway: topology-specific metrics are very close and noisy. `appearance` has the best skeleton Chamfer and width MAE, while `dt_updated_prompt` has the lowest component error. Endpoint/branch F1 values are too small to support a strong claim.

## Overall Step-2000 Conclusion

At the fixed final checkpoint, `appearance` is the strongest method overall because it wins FID, LPIPS, PSNR, SSIM, skeleton Chamfer, and width MAE. `dt_updated_prompt` is still the best on the old structural Chamfer metric, so it should remain the structure-control baseline.

The current `topology` and `topology_weighted` variants do not justify a claim of improvement at step 2000. If topology remains in the thesis, it should be reframed as an attempted structural encoding whose current form needs refinement, or combined with appearance/DT rather than used alone.
