# Retraining Comparison

This report compares the current active model with the newly extracted strong retraining backup.

## Decision

Keep the current active model:

`models/best_model.keras`

Reason: it has the best overall accuracy and weighted F1 score for the project demo.

## Active Model

| Item | Value |
|---|---:|
| Architecture | EfficientNetB0 |
| Test accuracy | 96.5204% |
| Weighted F1 | 96.50% |
| Macro F1 | 96.95% |
| Test images | 661 |
| Classes | 8 |

## Strong Retrain Backup

Backup path:

`/home/ron/ml/fdm_artifact_backups/expanded_fdm_results_strong`

| Item | Value |
|---|---:|
| Architecture | EfficientNetB3 |
| Test accuracy | 96.4589% |
| Weighted F1 | 96.44% |
| Macro F1 | 97.34% |
| Test images | 706 |
| Classes | 8 |

## Result

The EfficientNetB3 strong retrain improved macro F1 slightly and performed very well on smaller classes like Off_platform and Warping. However, it did not beat the active EfficientNetB0 model on the main project metric:

| Metric | Active B0 | Strong B3 | Winner |
|---|---:|---:|---|
| Test accuracy | 96.5204% | 96.4589% | Active B0 |
| Weighted F1 | 96.50% | 96.44% | Active B0 |
| Macro F1 | 96.95% | 97.34% | Strong B3 |

For the final presentation and Streamlit demo, use the active EfficientNetB0 model because it gives the best overall accuracy.

## Active Defect Classes

The current model can classify these 8 categories:

1. Cracking
2. Layer_shifting
3. No_defect
4. Off_platform
5. Spaghetti
6. Stringing
7. Under_extrusion
8. Warping

