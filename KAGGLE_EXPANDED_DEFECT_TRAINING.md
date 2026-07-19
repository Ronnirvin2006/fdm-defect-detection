# Kaggle Expanded Defect Training

This guide is for training more defect classes on Kaggle without filling laptop storage.

## What The Current Model Detects

The current trained model detects only:

- Cracking
- Layer_shifting
- Off_platform
- Stringing
- Warping

To detect more classes, attach additional datasets and retrain.

## Recommended Extra Datasets

Add these as Kaggle inputs where available:

1. Current base dataset

   `wengmhu/fdm-3d-printing-defect-dataset`

2. Kaggle 3D printing errors

   `nimbus200/3d-printing-errors`

   Useful labels: `GOOD`, `STRINGING`, `UNDEREXTRUSION`

3. Kaggle 3D printing defects

   `ssharkov/3d-printing-defects`

   Useful because it is exported in folder/annotation format and may help with extra defect appearances.

4. Kaggle 3D printer defected dataset

   `justin900429/3d-printer-defected-dataset`

   Useful for binary defected vs non-defected experiments.

Roboflow and Zenodo datasets can also be used, but they usually require manual download/export first and then adding them to Kaggle as a private dataset.

## Kaggle Notebook Setup

Start a Kaggle notebook with:

- Accelerator: GPU T4 x2 if available
- Internet: On
- Inputs: attach the datasets above

Clone the project:

```python
%cd /kaggle/working
!git clone https://github.com/Ronnirvin2006/fdm-defect-detection.git
%cd /kaggle/working/fdm-defect-detection
```

Do not run `pip install -r requirements.txt` unless Kaggle is missing packages. Kaggle already has TensorFlow.

Check TensorFlow GPU:

```python
!nvidia-smi
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

## Check Input Paths

Run:

```python
!find /kaggle/input -maxdepth 3 -type d | sort | head -100
```

Look for paths like:

```text
/kaggle/input/datasets
/kaggle/input/3d-printing-errors
/kaggle/input/3d-printing-defects
/kaggle/input/3d-printer-defected-dataset
```

## Prepare Expanded Dataset

Use the paths that exist in your notebook. Example:

```python
!python src/prepare_expanded_dataset.py \
  --sources /kaggle/input/datasets /kaggle/input/3d-printing-errors /kaggle/input/3d-printing-defects /kaggle/input/3d-printer-defected-dataset \
  --output /kaggle/working/expanded_fdm_dataset \
  --mode symlink \
  --min-images 20
```

If one path does not exist, remove it from the command.

Inspect the merged dataset:

```python
!FDM_DATA_RAW=/kaggle/working/expanded_fdm_dataset python src/inspect_dataset.py
```

## Train Expanded Model

Use EfficientNetB0 first:

```python
!FDM_DATA_RAW=/kaggle/working/expanded_fdm_dataset python src/train.py \
  --architecture efficientnet_b0 \
  --epochs 25 \
  --fine-tune-epochs 10 \
  --batch-size 32 \
  --mixed-precision
```

If GPU memory fails:

```python
!FDM_DATA_RAW=/kaggle/working/expanded_fdm_dataset python src/train.py \
  --architecture efficientnet_b0 \
  --epochs 20 \
  --fine-tune-epochs 5 \
  --batch-size 16 \
  --mixed-precision
```

## Evaluate Expanded Model

```python
!FDM_DATA_RAW=/kaggle/working/expanded_fdm_dataset python src/evaluate_model.py --batch-size 32
```

## Save Results

```python
!zip -r expanded_fdm_results.zip models outputs
```

Download `expanded_fdm_results.zip` from Kaggle output.

## Important Warning

Only claim classes that appear in the final merged dataset inspection. Do not claim under-extrusion, over-extrusion, nozzle clog, or no-defect unless they appear as class folders after running `prepare_expanded_dataset.py`.
