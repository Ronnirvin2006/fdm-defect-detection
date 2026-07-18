# Final Project Summary

## Title

AI-Based Real-Time Defect Detection in FDM 3D Printing Using Deep Learning

## Objective

The objective is to automate visual quality inspection in FDM 3D printing using a CNN-based image classifier. The system predicts the defect class from a camera image and provides confidence scores, evaluation metrics, visual explanation, and suggested corrective actions.

## Syllabus Alignment

- Unit I, Applications of AI: automated quality control in additive manufacturing.
- Unit V, Deep Learning and CNN: transfer learning, CNN feature extraction, image classification, evaluation, and explainability.

## Dataset Used

Dataset: FDM 3D Printing Defect Dataset from Kaggle

Classes:

- Cracking
- Layer_shifting
- Off_platform
- Stringing
- Warping

Dataset inspection:

- Total images: 1912
- Train images: 1338
- Validation images: 287
- Test images: 287
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

Test accuracy: 97.56%

Per-class F1-score:

- Cracking: 0.9640
- Layer_shifting: 0.9643
- Off_platform: 0.9286
- Stringing: 1.0000
- Warping: 0.9814

Weighted F1-score: 0.9758

Macro ROC-AUC: 0.9995

Local CPU inference speed:

- Average time per image: 64.24 ms
- Estimated throughput: 15.57 FPS

## Project Outputs

- Trained Keras model: `models/best_model.keras`
- Class labels: `models/class_names.json`
- Metrics: `outputs/reports/metrics.json`
- Classification report: `outputs/reports/classification_report.txt`
- Confusion matrix: `outputs/figures/confusion_matrix.png`
- Training curves: `outputs/figures/training_curves.png`
- ROC curves: `outputs/figures/roc_curves.png`
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

The current trained model only detects the five classes used during training. More defects such as under-extrusion, over-extrusion, nozzle clogging, blobs/zits, layer separation, and no-defect require additional labeled data and retraining.

## Future Work

- Retrain using expanded Roboflow, Kaggle, and Zenodo datasets.
- Add more model comparisons such as MobileNetV3, ResNet50, and Custom CNN.
- Add true continuous webcam/video inference.
- Add closed-loop control suggestions that can be sent to a printer controller after human approval.
