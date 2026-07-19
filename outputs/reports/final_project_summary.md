# Final Project Summary

## Title

AI-Based Real-Time Defect Detection in FDM 3D Printing Using Deep Learning

## Objective

The objective is to automate visual quality inspection in FDM 3D printing using a CNN-based image classifier. The system predicts the defect class from a camera image and provides confidence scores, evaluation metrics, visual explanation, and suggested corrective actions.

## Syllabus Alignment

- Unit I, Applications of AI: automated quality control in additive manufacturing.
- Unit V, Deep Learning and CNN: transfer learning, CNN feature extraction, image classification, evaluation, and explainability.

## Dataset Used

Dataset: Expanded FDM defect dataset merged from Kaggle sources

Classes:

- Cracking
- Layer_shifting
- No_defect
- Off_platform
- Spaghetti
- Stringing
- Under_extrusion
- Warping

Dataset inspection:

- Total images: 4405
- Train images: 3083
- Validation images: 661
- Test images: 661
- Corrupt images: 0

## Model

Architecture: EfficientNetB0 transfer learning

Important techniques:

- 224 x 224 image resizing
- ImageNet transfer learning
- Data augmentation
- Class weighting for imbalance
- Fine-tuning
- Test-set evaluation

## Results

Test accuracy: 96.52%

Per-class F1-score:

- Cracking: 0.9714
- Layer_shifting: 0.9815
- No_defect: 0.9380
- Off_platform: 0.9655
- Spaghetti: 0.9863
- Stringing: 0.9890
- Under_extrusion: 0.9424
- Warping: 0.9816

Macro F1-score: 0.9695

Weighted F1-score: 0.9650

## Project Outputs

- Trained Keras model: `models/best_model.keras`
- Class labels: `models/class_names.json`
- Metrics: `outputs/reports/metrics.json`
- Classification report: `outputs/reports/classification_report.txt`
- Confusion matrix: `outputs/figures/confusion_matrix.png`
- Training curves: `outputs/figures/training_curves.png`
- Grad-CAM output: `outputs/figures/gradcam_<image_name>.png`

## Added Research-Gap Features

The base research gap says many systems stop at classification. This project extends the classifier with:

- Confidence score display
- Root-cause suggestions
- Corrective-action recommendations
- Grad-CAM explainability
- Interactive upload and camera-snapshot panel
- Dataset expansion plan for more defect classes

## Current Limitation

The current trained model detects the eight classes used during expanded training. More defects such as over-extrusion, nozzle clogging, blobs/zits, and layer separation require additional labeled data and retraining.

## Future Work

- Retrain using expanded Roboflow, Kaggle, and Zenodo datasets.
- Add more model comparisons such as MobileNetV3, ResNet50, and Custom CNN.
- Add true continuous webcam/video inference.
- Add closed-loop control suggestions that can be sent to a printer controller after human approval.
