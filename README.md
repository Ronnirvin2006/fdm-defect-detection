# Automated Defect Detection in FDM 3D Printing using CNNs

TensorFlow project for classifying camera images of FDM 3D printed parts into defect categories using the Kaggle FDM 3D Printing Defect Dataset.

## Goal

- Train a CNN-based image classifier for FDM print defects.
- Use transfer learning with MobileNetV2 for stronger accuracy on a weak laptop.
- Save a reusable model and generate project outputs: metrics, confusion matrix, classification report, and prediction results.

## Dataset

Kaggle dataset: `wengmhu/fdm-3d-printing-defect-dataset`

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
python src/train.py --epochs 25 --fine-tune-epochs 10 --batch-size 16 --mixed-precision
```

If GPU memory is low:

```bash
python src/train.py --epochs 25 --fine-tune-epochs 5 --batch-size 8
```

Predict one image:

```bash
python src/predict.py /path/to/image.jpg
```

## Expected Outputs

- `models/best_model.keras`
- `models/class_names.json`
- `outputs/reports/dataset_inspection.md`
- `outputs/reports/metrics.json`
- `outputs/reports/classification_report.txt`
- `outputs/figures/training_curves.png`
- `outputs/figures/confusion_matrix.png`

## Research Alignment

- Unit V: CNN, convolution layers, transfer learning, classification.
- Unit I: AI application in automated manufacturing quality control.
