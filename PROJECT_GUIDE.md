# FDM Defect Detection Project Guide

## 1. Project Purpose

This project performs automated visual quality inspection for Fused Deposition
Modeling (FDM) 3D printing. A TensorFlow convolutional neural network receives a
camera image, identifies the most likely print condition, reports confidence, and
provides possible causes and corrective actions.

The active model recognizes eight classes:

1. `Cracking`
2. `Layer_shifting`
3. `No_defect`
4. `Off_platform`
5. `Spaghetti`
6. `Stringing`
7. `Under_extrusion`
8. `Warping`

This aligns with AI applications in automated manufacturing quality control and
with CNN topics including image preprocessing, transfer learning, classification,
fine-tuning, evaluation, and explainability.

## 2. System Flow

```text
Dataset images
    -> dataset inspection and class-folder labeling
    -> stratified train/validation/test split
    -> augmentation and class weighting
    -> EfficientNetB0 transfer learning and fine-tuning
    -> saved Keras model
    -> uploaded image, snapshot, or OpenCV camera frame
    -> resize to 224 x 224 RGB
    -> eight softmax confidence scores
    -> predicted class, risk level, causes, and corrective actions
```

The continuous dashboard averages the last five predictions to reduce label
flicker. Predictions below 60% confidence are marked `Uncertain`.

## 3. Dataset and Model

The final expanded dataset combines compatible labeled Kaggle sources. Labels are
provided by class folders or dataset annotation files, not by image filenames.

| Item | Value |
|---|---:|
| Total images | 4,405 |
| Training images | 3,083 |
| Validation images | 661 |
| Test images | 661 |
| Corrupt images | 0 |
| Input size | 224 x 224 RGB |
| Architecture | EfficientNetB0 |
| Test accuracy | 96.52% |
| Weighted F1 | 96.50% |
| Macro F1 | 96.95% |

The model uses ImageNet transfer learning, image augmentation, class weights for
imbalanced classes, early stopping, learning-rate reduction, and a second
fine-tuning stage.

## 4. Launch and Use

### Recommended Continuous Camera Dashboard

```bash
cd /home/ron/ml/fdm_defect_detection
.venv/bin/python src/live_dashboard.py
```

Open `http://127.0.0.1:8765` in a browser. Camera device `0` is used by default.
The camera opens when the live video connects and is released automatically when
the page closes. Stop the entire server at any time with `Ctrl+C` in its terminal.
To use camera device `1`:

```bash
FDM_CAMERA_INDEX=1 .venv/bin/python src/live_dashboard.py
```

The dashboard continuously displays the camera, condition, confidence, risk,
possible causes, corrective actions, axis checks, and nozzle-cleaning checklist.

### Upload or Camera Snapshot Interface

```bash
cd /home/ron/ml/fdm_defect_detection
.venv/bin/streamlit run src/app.py
```

Open `http://127.0.0.1:8501`. Use `Upload Image` for a saved image or `Camera
Snapshot` for a single camera capture. Streamlit WebRTC is disabled by default on
this laptop because its native package is unstable.

### Command-Line Prediction

```bash
cd /home/ron/ml/fdm_defect_detection
.venv/bin/python src/predict.py /absolute/path/to/image.jpg
```

### External Dataset Validation

Use a genuinely separate labeled dataset organized as one folder per class:

```bash
.venv/bin/python src/validate_external.py /path/to/external_class_folders
```

This writes external accuracy, per-class metrics, and a confusion matrix. Do not use
the original training images for this command when claiming external validity.

### Optional OctoPrint Telemetry

```bash
FDM_OCTOPRINT_URL=http://octoprint.local \
FDM_OCTOPRINT_API_KEY=your_private_key \
.venv/bin/python src/live_dashboard.py
```

This provides real printer state, nozzle/bed temperatures, print progress, and
remaining time. Standard OctoPrint status does not provide current X/Y/Z positions
or actual motion speed, so the dashboard reports those as unavailable. Axis advice
is a visual inspection recommendation, not automatic movement. Nozzle-cleaning
status is explicitly an operator confirmation, not a sensor reading.

## 5. File and Folder Map

### Root Files

| Path | Purpose |
|---|---|
| `README.md` | Short project overview, common commands, results, and demo instructions. |
| `PROJECT_GUIDE.md` | Complete explanation of the project, workflow, launching, and file structure. |
| `requirements.txt` | Exact Python packages required by training, evaluation, demos, and presentation generation. |
| `.gitignore` | Prevents environments, credentials, large datasets, caches, logs, and temporary results from entering Git. |
| `COLAB_OR_KAGGLE_TRAINING.md` | Basic instructions for training on Kaggle, Colab, or the laptop. |
| `KAGGLE_EXPANDED_DEFECT_TRAINING.md` | Instructions for combining multiple labeled datasets and training the eight-class model. |
| `STRONG_RETRAINING_GUIDE.md` | Longer EfficientNet retraining configurations and model-selection rules. |

### Source Code

| Path | Purpose |
|---|---|
| `src/config.py` | Central paths, image size, split ratios, batch defaults, random seed, and writable directories. |
| `src/download_dataset.py` | Downloads the original Kaggle dataset using the Kaggle command-line API. |
| `src/download_dataset_kagglehub.py` | Downloads using KaggleHub and copies data into the project without duplicating an existing cache. |
| `src/inspect_dataset.py` | Finds class folders, counts images and dimensions, checks corruption, and writes the inspection report. |
| `src/prepare_expanded_dataset.py` | Reads folders, CSV labels, and text labels from multiple datasets; normalizes class names and creates one merged class-folder dataset. |
| `src/train.py` | Builds, trains, fine-tunes, evaluates, and saves EfficientNet or MobileNet classifiers. |
| `src/evaluate_model.py` | Re-evaluates the saved model and generates detailed metrics such as ROC-AUC and inference timing. |
| `src/validate_external.py` | Evaluates the active model on a genuinely separate class-folder dataset without retraining or re-splitting it. |
| `src/predict.py` | Runs one saved image through the active model and prints ranked confidence scores and recommendations. |
| `src/gradcam.py` | Produces a Grad-CAM heatmap showing which image regions influenced a prediction. |
| `src/defect_knowledge.py` | Stores descriptions, likely causes, and corrective actions for known FDM defects. |
| `src/app.py` | Streamlit upload and camera-snapshot interface; optional WebRTC code remains disabled by default. |
| `src/live_dashboard.py` | Stable OpenCV continuous camera server, temporal prediction averaging, feedback dashboard, cleaning checklist, and optional OctoPrint telemetry. |

### Model Files

| Path | Purpose |
|---|---|
| `models/best_model.keras` | Active trained EfficientNetB0 model used by every prediction interface. |
| `models/class_names.json` | Class index order required to translate model outputs into defect names. |
| `models/.gitkeep` | Keeps the directory present if model artifacts are removed. |

### Generated Outputs

| Path | Purpose |
|---|---|
| `outputs/reports/metrics.json` | Machine-readable dataset split, architecture, class weights, loss, and accuracy. |
| `outputs/reports/classification_report.txt` | Precision, recall, F1-score, and support for every class. |
| `outputs/reports/dataset_inspection.md` | Dataset class counts, image sizes, and corruption check. |
| `outputs/reports/final_project_summary.md` | Presentation-ready objective, methods, results, limitations, and future work. |
| `outputs/reports/research_literature.md` | Literature notes supporting the project approach. |
| `outputs/reports/dataset_expansion_research.md` | Research and mapping notes for additional datasets and defect classes. |
| `outputs/reports/retraining_comparison.md` | Comparison that explains why EfficientNetB0 remains active instead of the B3 backup. |
| `outputs/figures/training_curves.png` | Training/validation accuracy and loss over epochs. |
| `outputs/figures/confusion_matrix.png` | True-versus-predicted class comparison. |
| `outputs/figures/gradcam_*.png` | Visual explanation overlays for sample predictions. |
| `outputs/presentation/FDM_Defect_Detection_Project.pptx` | Editable 15-slide project presentation. |
| `outputs/paper/FDM_Defect_Detection_Conference_Paper.md` | Version-controlled source for the IEEE-style conference manuscript. |
| `outputs/paper/FDM_Defect_Detection_Conference_Paper.docx` | Editable two-column manuscript draft. |
| `outputs/paper/FDM_Defect_Detection_Conference_Paper.pdf` | PDF manuscript draft for review. |
| `outputs/paper/SUBMISSION_CHECKLIST.md` | Required metadata, template, validation, authorship, and licensing checks before conference submission. |
| `outputs/reports/project_completion_audit.md` | Final evidence of completed deliverables and the one data-dependent validation gap. |
| `outputs/predictions/.gitkeep` | Placeholder for future saved prediction results. |

### Supporting Folders

| Path | Purpose |
|---|---|
| `data/raw/` | Original dataset files; intentionally excluded from Git because they are large. |
| `data/processed/` | Prepared or merged datasets; intentionally excluded from Git. |
| `logs/` | Training logs and checkpoints; generated content is excluded from Git. |
| `tools/create_presentation.py` | Rebuilds the PowerPoint from current project metrics and figures. |
| `tools/create_conference_paper.py` | Rebuilds the editable conference DOCX from the manuscript source and project figures. |
| `.venv/` | Local Python environment; never pushed to Git. |
| `.keras/` | Local Keras model/download cache; never pushed to Git. |
| `.kaggle/` | Local Kaggle credentials/configuration; never pushed to Git. |

## 6. Important Limitations

- The classifier recognizes only the eight classes used during training.
- Accuracy on the held-out dataset does not guarantee identical performance under
  a new camera angle, printer, filament color, lighting condition, or background.
- Low-confidence predictions require human inspection.
- The system provides recommendations but does not automatically stop a printer,
  move axes, clean a nozzle, or change heater settings.
- Additional labeled data and retraining are required for defects such as nozzle
  clog, over-extrusion, blobs/zits, layer separation, and Z-banding.
