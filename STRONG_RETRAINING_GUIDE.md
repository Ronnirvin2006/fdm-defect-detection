# Strong Kaggle Retraining Guide

Current best expanded model:

- Classes: 8
- Test accuracy: 96.52%
- Weighted F1-score: 96.50%
- Architecture: EfficientNetB0

Retraining can improve the result, but it is not guaranteed. Keep a new run only if it beats the current best.

## Prepare Dataset

Use the same balanced 8-class dataset:

```python
%cd /kaggle/working/fdm-defect-detection
!git pull

!rm -rf /kaggle/working/expanded_fdm_dataset
!python src/prepare_expanded_dataset.py \
  --sources /kaggle/input/datasets \
  --output /kaggle/working/expanded_fdm_dataset \
  --mode symlink \
  --min-images 20 \
  --max-per-class 1000

!rm -rf /kaggle/working/expanded_fdm_dataset/Defected
!FDM_DATA_RAW=/kaggle/working/expanded_fdm_dataset python src/inspect_dataset.py
```

## Best First Long Run

This is the recommended serious retrain:

```python
!FDM_DATA_RAW=/kaggle/working/expanded_fdm_dataset python src/train.py \
  --architecture efficientnet_b1 \
  --epochs 45 \
  --fine-tune-epochs 18 \
  --batch-size 24 \
  --dropout 0.35 \
  --augmentation strong \
  --initial-lr 0.0008 \
  --fine-tune-lr 0.000008 \
  --patience 8 \
  --lr-patience 3 \
  --mixed-precision
```

## If GPU Memory Fails

Use EfficientNetB0:

```python
!FDM_DATA_RAW=/kaggle/working/expanded_fdm_dataset python src/train.py \
  --architecture efficientnet_b0 \
  --epochs 45 \
  --fine-tune-epochs 18 \
  --batch-size 32 \
  --dropout 0.35 \
  --augmentation strong \
  --initial-lr 0.0008 \
  --fine-tune-lr 0.000008 \
  --patience 8 \
  --lr-patience 3 \
  --mixed-precision
```

## Experimental Stronger Run

Only try this if T4 memory is fine and you have time:

```python
!FDM_DATA_RAW=/kaggle/working/expanded_fdm_dataset python src/train.py \
  --architecture efficientnet_b3 \
  --epochs 50 \
  --fine-tune-epochs 20 \
  --batch-size 16 \
  --dropout 0.4 \
  --augmentation strong \
  --initial-lr 0.0006 \
  --fine-tune-lr 0.000005 \
  --patience 10 \
  --lr-patience 3 \
  --mixed-precision
```

## Save Immediately

After training:

```python
!zip -r expanded_fdm_results_strong.zip models outputs
```

Download the zip and click **Save Version** in Kaggle.

## Keep Or Reject

Keep the new model only if:

- Accuracy is higher than 96.52%, or
- Weighted F1-score is higher than 96.50%, or
- Important weak classes such as `No_defect` and `Under_extrusion` improve.

Otherwise keep the current model.
