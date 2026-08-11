# Research Notes: Automated Defect Detection in FDM 3D Printing using CNNs

## Project Topic

This project applies convolutional neural networks to camera images of FDM/FFF 3D printed parts. The model classifies images into print-quality or defect categories so that visual inspection can be automated during or after printing.

## Syllabus Alignment

- Unit I, Applications of AI: automated quality inspection in additive manufacturing.
- Unit V, Deep Learning and CNN: convolutional layers, image classification, transfer learning, evaluation metrics.

## Dataset

- Dataset: FDM 3D Printing Defect Dataset
- Source: Kaggle, `wengmhu/fdm-3d-printing-defect-dataset`
- Public listing: https://www.kaggle.com/datasets/wengmhu/fdm-3d-printing-defect-dataset
- Search listing reports 1912 images and five defect image categories.

## Key Research Papers and Sources

1. Defect detection in fused deposition modelling using lightweight CNN models

   Source: https://www.sciencedirect.com/science/article/abs/pii/S0952197624019614

   Relevance: This is closely aligned with the project because it uses real-time captured FDM images, resizes them to 224 x 224, applies augmentation, and uses CNN-based classification for defects such as over-deposition, staircase effects, and voids.

2. Real-Time 3D Printing Remote Defect Detection (Stringing) with Computer Vision and Artificial Intelligence

   Source: https://www.mdpi.com/2227-9717/8/11/1464

   Relevance: Focuses on real-time camera/video monitoring of FFF/FDM printing and stringing defect detection. It supports the idea that visual AI can monitor print quality without manual inspection.

3. A Survey of Image-Based Fault Monitoring in Additive Manufacturing

   Source: https://www.mdpi.com/1424-8220/23/15/6821

   Relevance: Good background source for the literature review. It explains how image-based monitoring is used across additive manufacturing and why vision systems are important for fault detection.

4. Generalisable 3D Printing Error Detection and Correction via Multi-Head Neural Networks

   Source: https://www.nature.com/articles/s41467-022-31985-y

   Relevance: Strong source for explaining why automated defect detection matters. It discusses the need for generalizable monitoring because human visual inspection is not continuous and print errors waste time and material.

5. Real-time remote monitoring and defect detection in smart additive manufacturing using CNNs

   Source: https://www.sciencedirect.com/science/article/abs/pii/S0263224125007213

   Relevance: Uses a CNN model on thousands of experimental images for stringing defect detection and compares a custom CNN with pre-trained models. This supports our choice to use transfer learning and compare model outputs using accuracy and model size.

6. Convolutional Neural Network-Based Defect Detection Technique in FDM 3D Printers

   Source: https://research.nu.edu.kz/en/publications/convolutional-neural-network-based-defect-detection-technique-in-/

   Relevance: Uses CNNs for FDM anomaly detection with experimentally created defects such as over-extrusion and layer shift. This supports the manufacturing-specific CNN framing.

## Why CNN Is Suitable

CNNs are suitable because FDM defects have visible spatial patterns:

- Warping: corners or edges lift from the bed.
- Stringing: thin filament strands appear between separated regions.
- Under-extrusion: missing lines, gaps, weak layers, or inconsistent filament flow.
- Healthy/good print: clean boundaries and consistent deposition.

Convolution filters can learn edges, textures, gaps, strands, and surface irregularities directly from image pixels.

## Planned Model Outputs

The trained model can produce:

- predicted defect class for an image
- confidence score for each class
- training and validation accuracy
- test accuracy
- precision, recall, and F1-score per class
- confusion matrix
- training loss/accuracy curves
- sample image predictions
- saved TensorFlow model for later demo or deployment

## Practical Model Choice

The final active model is EfficientNetB0. It provided the strongest overall test accuracy and weighted F1 score among the retained project runs while remaining small enough for local CPU inference. A stronger EfficientNetB3 retrain improved macro F1 slightly but did not beat EfficientNetB0 on accuracy or weighted F1.

## Current Literature Position

- Hu et al. (2024) reported a lightweight improved YOLOv8 system for five FFF defects with 97.5% mAP50. This is an object-detection study, whereas this project performs image-level eight-class classification and operator feedback.
- Kim et al. (2022) demonstrated that transfer learning and Grad-CAM are feasible for small-data spaghetti-failure monitoring, reporting 94% accuracy.
- Aktepe and Ergün (2026) compared EfficientNetB0 and MobileNetV2 on the original 1,912-image Kaggle dataset. The present project extends the data into eight classes and adds continuous monitoring, uncertainty handling, explainability, and optional telemetry.
- Kuriachen et al. (2025) highlighted lightweight CNNs for real-time multi-class FDM monitoring and the need to improve generalization.

Primary references and DOIs are included in `outputs/paper/FDM_Defect_Detection_Conference_Paper.md`.

## Limitation

High accuracy on the dataset does not automatically prove performance on a new printer, recording session, camera angle, or lighting condition. The reported 96.52% is based on a stratified image-level held-out split. A future publication-quality extension should use group-aware splitting by print or recording session and an external printer dataset.
