# Project Completion Audit

Date: 2026-08-11

## Completed

- Active eight-class EfficientNetB0 model and class-index mapping are present.
- Held-out image-level evaluation reports 96.52% accuracy and 96.50% weighted F1.
- Confusion matrix, training curves, Grad-CAM, per-class report, and model comparison are present.
- Streamlit upload/snapshot interface is available.
- Stable OpenCV continuous dashboard includes temporal averaging and uncertainty handling.
- Camera is opened only while a video client is connected and is released when the page closes.
- Defect-specific causes, corrective actions, axis checks, and nozzle-cleaning confirmation are available.
- Optional OctoPrint state, temperature, progress, and remaining-time integration is implemented.
- CPU benchmark is reproducible through `src/benchmark_model.py`.
- External labeled datasets can be evaluated without retraining through `src/validate_external.py`.
- The 15-slide presentation was updated to the OpenCV dashboard and visually inspected.
- IEEE-style manuscript source, editable DOCX, PDF, generator, and submission checklist were produced.
- Python source and tool files compile successfully.

## Evidence

- Dataset: 4,405 images; 3,083 train; 661 validation; 661 test; zero corrupt.
- Test accuracy: 0.9652042389.
- Macro F1: 0.9695.
- Weighted F1: 0.9650.
- Active model parameters: 4,059,819.
- Active model size: 41,227,725 bytes.
- Local CPU benchmark on 2026-08-11: approximately 108.3 ms per image across 30 timed runs after warm-up.

## Data-Dependent Validation Gap

No genuinely unseen labeled external dataset is currently stored locally. The
independent Zenodo record 14712897 was inspected, but its test archive alone is
12.63 GB and was not downloaded after disk cleanup. Therefore, no external accuracy
is claimed. Use:

```bash
.venv/bin/python src/validate_external.py /path/to/external_class_folders
```

Only images that were not used for training or model selection qualify as external
validation. A recording-grouped retrain and multi-printer test remain research
extensions, not hidden incomplete software work.

## Storage Cleanup

Approximately 10 GB was reclaimed by deleting two confirmed duplicate archives from
Trash, generated pip/browser/development caches, a generated VS Code symbol database,
temporary previews, and an ignored duplicate model. The active virtual environment,
model, labels, project outputs, artifact backups, Gazebo/ROS data, browser profile,
VS Code extensions, and personal files were preserved.
