# Training on Google Colab or Kaggle

This project can train locally, but Colab or Kaggle GPU is better than the GTX 1650 setup currently detected by TensorFlow on this laptop.

## Can Codex Train Directly on Colab?

No. Codex in this terminal cannot log into your Google Colab or Kaggle account and run the notebook for you. You must open Colab/Kaggle and run the commands there. The project code is ready for that.

## Important Dataset Point

The images are not labelled inside the filename. They are labelled by folder:

```text
data/Stringing/*.jpg        -> Stringing
data/Warping/*.jpg          -> Warping
data/Layer_shifting/*.jpg   -> Layer_shifting
data/Off_platform/*.jpg     -> Off_platform
data/Cracking/*.jpg         -> Cracking
```

This is normal for image classification.

## Recommended: Kaggle Notebook

Kaggle is easiest because the dataset is already from Kaggle.

1. Open Kaggle.
2. Create a new Notebook.
3. Click **Add Input**.
4. Add dataset: `wengmhu/fdm-3d-printing-defect-dataset`.
5. Turn on GPU from notebook settings.
6. Upload this project folder, or copy the `src` folder files.
Expected Kaggle dataset path is usually similar to:

```text
/kaggle/input/fdm-3d-printing-defect-dataset/FDM-3D-Printing-Defect-Dataset/data
```

Run:

```bash
export FDM_DATA_RAW=/kaggle/input/fdm-3d-printing-defect-dataset
python src/inspect_dataset.py
python src/train.py --architecture efficientnet_b0 --epochs 25 --fine-tune-epochs 10 --batch-size 32 --mixed-precision
```

If GPU memory is low:

```bash
export FDM_DATA_RAW=/kaggle/input/fdm-3d-printing-defect-dataset
python src/train.py --architecture efficientnet_b0 --epochs 20 --fine-tune-epochs 5 --batch-size 16 --mixed-precision
```

## Google Colab Option

In Colab, enable GPU:

```text
Runtime -> Change runtime type -> T4 GPU
```

Then install packages:

```bash
pip install tensorflow==2.16.1 numpy==1.26.4 pillow==10.4.0 matplotlib==3.9.2 scikit-learn==1.5.1 pandas==2.2.2
```

Upload the project folder and dataset to Google Drive, then mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Example command after copying the project to Colab:

```bash
cd /content/drive/MyDrive/fdm_defect_detection
export FDM_DATA_RAW=/content/drive/MyDrive/fdm_defect_detection/data/raw
python src/inspect_dataset.py
python src/train.py --architecture efficientnet_b0 --epochs 25 --fine-tune-epochs 10 --batch-size 32 --mixed-precision
```

## Laptop Fallback

Use this if training on your laptop:

```bash
cd /home/ron/ml/fdm_defect_detection
source .venv/bin/activate
python src/train.py --architecture mobilenet_v2 --epochs 15 --fine-tune-epochs 5 --batch-size 8
```

Do not use `--mixed-precision` locally unless TensorFlow shows a GPU in:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## After Training

Send Codex these files:

```text
models/best_model.keras
models/class_names.json
outputs/reports/metrics.json
outputs/reports/classification_report.txt
outputs/figures/confusion_matrix.png
outputs/figures/training_curves.png
```

Then Codex can check the metrics and help with prediction verification.
