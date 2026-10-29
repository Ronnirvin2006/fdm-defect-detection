# Local Model Comparison

Same grouped validation split; reused for model selection. Prior-model training overlap is unknown. These are not independent test results.

| Metric | Original | Candidate |
|---|---:|---:|
| accuracy | 51.84% | 83.10% |
| macro_f1 | 57.62% | 84.98% |
| weighted_f1 | 53.86% | 82.98% |

## Largest Candidate Confusions

- No_defect predicted as Under_extrusion: 292 images.
- No_defect predicted as Stringing: 252 images.
- Under_extrusion predicted as No_defect: 91 images.
- No_defect predicted as Spaghetti: 23 images.
- Under_extrusion predicted as Stringing: 16 images.
- Warping predicted as Cracking: 15 images.
- Cracking predicted as Warping: 12 images.
- Spaghetti predicted as Under_extrusion: 9 images.

## Decision

Candidate is ready for controlled comparison on new print photos/videos. Keep the active model until external checks pass. Do not compare these scores directly with the historical 96.52% test result.
