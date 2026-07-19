# Automated Defect Detection in FDM 3D Printing using CNNs

TensorFlow project for classifying camera images of FDM 3D printed parts into defect categories using the Kaggle FDM 3D Printing Defect Dataset.

## Goal

- Train a CNN-based image classifier for FDM print defects.
- Use transfer learning with EfficientNetB0 for strong accuracy and MobileNetV2 as a lighter fallback.
- Save a reusable model and generate project outputs: metrics, confusion matrix, ROC curves, Grad-CAM, classification report, recommendations, and prediction results.

## Dataset

Kaggle dataset: `wengmhu/fdm-3d-printing-defect-dataset`

For training more defect classes, see:

```text
KAGGLE_EXPANDED_DEFECT_TRAINING.md
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

Generate a Grad-CAM explanation image:

```bash
python src/gradcam.py /path/to/image.jpg
```

Launch the interactive upload/camera panel:

```bash
streamlit run src/app.py
```

## Expected Outputs

- `models/best_model.keras`
- `models/class_names.json`
- `outputs/reports/dataset_inspection.md`
- `outputs/reports/metrics.json`
- `outputs/reports/classification_report.txt`
- `outputs/figures/training_curves.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/roc_curves.png`
- `outputs/figures/gradcam_<image_name>.png`
- `outputs/reports/evaluation_metrics.json`
- `outputs/reports/dataset_expansion_research.md`
- `outputs/reports/final_project_summary.md`

## Current Trained Result

- Architecture: EfficientNetB0
- Test accuracy: 97.56%
- Weighted F1-score: 97.58%
- Macro ROC-AUC: 99.95%
- Local CPU inference estimate: about 64 ms per image, around 15 FPS

## Interactive Demo

The Streamlit panel supports:

- Uploading an image.
- Taking a camera snapshot.
- Displaying predicted defect class and confidence.
- Showing possible causes and corrective actions.

The current trained model detects only the five classes in the Kaggle dataset. Extra defects such as under-extrusion, over-extrusion, nozzle clog, blobs/zits, and no-defect require additional labeled datasets and retraining.

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
