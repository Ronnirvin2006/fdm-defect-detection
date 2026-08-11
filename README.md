# Automated Defect Detection in FDM 3D Printing using CNNs

TensorFlow project for classifying camera images of FDM 3D printed parts into defect categories using the Kaggle FDM 3D Printing Defect Dataset.

For the complete architecture, launch guide, and file-by-file explanation, see
`PROJECT_GUIDE.md`.

## Goal

- Train a CNN-based image classifier for FDM print defects.
- Use transfer learning with EfficientNetB0 for strong accuracy and MobileNetV2 as a lighter fallback.
- Save a reusable model and generate project outputs: metrics, confusion matrix, Grad-CAM, classification report, recommendations, and prediction results.

## Dataset

Kaggle dataset: `wengmhu/fdm-3d-printing-defect-dataset`

For training more defect classes, see:

```text
KAGGLE_EXPANDED_DEFECT_TRAINING.md
```

For a serious accuracy-improvement retrain, see:

```text
STRONG_RETRAINING_GUIDE.md
```

## Commands

Install dependencies after Python pip/venv is available:

```bash
cd /home/ron/ml/fdm_defect_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and inspect:

```bash
python src/download_dataset.py
python src/inspect_dataset.py
```

Kaggle needs `kaggle.json` at:

```text
/home/ron/ml/fdm_defect_detection/.kaggle/kaggle.json
```

Train:

```bash
python src/train.py --architecture efficientnet_b0 --epochs 25 --fine-tune-epochs 10 --batch-size 32 --mixed-precision
```

If GPU memory is low:

```bash
python src/train.py --epochs 25 --fine-tune-epochs 5 --batch-size 8
```

Predict one image:

```bash
python src/predict.py /path/to/image.jpg
```

Run full evaluation with ROC-AUC and inference-time measurement:

```bash
python src/evaluate_model.py --batch-size 32
```

Validate against a genuinely separate class-folder dataset:

```bash
.venv/bin/python src/validate_external.py /path/to/external_class_folders
```

Generate a Grad-CAM explanation image:

```bash
python src/gradcam.py /path/to/image.jpg
```

Launch the interactive upload/live-webcam panel:

```bash
streamlit run src/app.py
```

Launch the stable continuous OpenCV dashboard (recommended for local live monitoring):

```bash
.venv/bin/python src/live_dashboard.py
```

Open `http://127.0.0.1:8765`. This reads camera device `0`, continuously runs the
classifier, overlays the detected class, and shows AI feedback in the browser.
The camera is released automatically when the page closes. Use another camera with
`FDM_CAMERA_INDEX=1`, or stop the server completely with `Ctrl+C`.

Optional real OctoPrint telemetry:

```bash
FDM_OCTOPRINT_URL=http://octoprint.local \
FDM_OCTOPRINT_API_KEY=your_key_here \
.venv/bin/python src/live_dashboard.py
```

Nozzle and bed temperatures, print state, progress, and time remaining come from
OctoPrint. Live XYZ positions and actual motion speed are displayed as unavailable
because the standard OctoPrint status API does not provide them. The dashboard never
automatically moves an axis or changes a heater setting.

## Expected Outputs

- `models/best_model.keras`
- `models/class_names.json`
- `outputs/reports/dataset_inspection.md`
- `outputs/reports/metrics.json`
- `outputs/reports/classification_report.txt`
- `outputs/figures/training_curves.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/gradcam_<image_name>.png`
- `outputs/reports/dataset_expansion_research.md`
- `outputs/reports/final_project_summary.md`
- `outputs/paper/FDM_Defect_Detection_Conference_Paper.docx`
- `outputs/paper/FDM_Defect_Detection_Conference_Paper.pdf`

## Current Trained Result

- Architecture: EfficientNetB0
- Classes: 8
- Test accuracy: 96.52%
- Weighted F1-score: 96.50%
- Macro F1-score: 96.95%

## Interactive Demo

The Streamlit panel supports:

- Uploading an image.
- Taking a camera snapshot as a fallback.
- Displaying predicted defect class and confidence.
- Showing AI feedback, possible causes, and corrective actions.

During presentation, use the upload tab for a known test image or run the separate
OpenCV dashboard for continuous real-time monitoring. Streamlit's native WebRTC live
tab is disabled by default on this laptop because that dependency is unstable here.

For a continuous local camera feed, prefer `src/live_dashboard.py`. It avoids the
native WebRTC component that is unstable on this laptop. The nozzle-cleaning value is
an explicitly marked operator confirmation, not a value inferred by the camera.

The current trained model detects `Cracking`, `Layer_shifting`, `No_defect`, `Off_platform`, `Spaghetti`, `Stringing`, `Under_extrusion`, and `Warping`. Extra defects such as over-extrusion, nozzle clog, blobs/zits, and layer separation require additional labeled datasets and retraining.

## Expanded Dataset Training

Use `src/prepare_expanded_dataset.py` on Kaggle to merge multiple attached datasets into one folder-per-class dataset:

```bash
python src/prepare_expanded_dataset.py --sources /kaggle/input/datasets /kaggle/input/3d-printing-errors --output /kaggle/working/expanded_fdm_dataset
```

Then train with:

```bash
FDM_DATA_RAW=/kaggle/working/expanded_fdm_dataset python src/train.py --architecture efficientnet_b0 --epochs 25 --fine-tune-epochs 10 --batch-size 32 --mixed-precision
```

## Research Alignment

- Unit V: CNN, convolution layers, transfer learning, classification.
- Unit I: AI application in automated manufacturing quality control.
