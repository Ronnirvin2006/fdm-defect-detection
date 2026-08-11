# An Explainable Real-Time Vision System for Multi-Class Defect Detection in FDM 3D Printing Using EfficientNetB0

**Ron Nirvin**  
Department of [Department Name], [Institution Name], [City, Country]  
[author.email@example.com]

> Submission draft: replace all bracketed metadata and apply the exact template supplied by the target conference before submission.

## Abstract

Fused deposition modeling (FDM) is accessible and economical, but printing failures can consume material, machine time, and operator effort before they are discovered. This paper presents an explainable visual monitoring system that classifies eight FDM print conditions: cracking, layer shifting, no defect, off-platform printing, spaghetti failure, stringing, under-extrusion, and warping. An expanded dataset of 4,405 labeled images was assembled from compatible public FDM image sources and divided by stratified image-level sampling into 3,083 training, 661 validation, and 661 test images. The proposed pipeline uses an ImageNet-pretrained EfficientNetB0 backbone, augmentation, class weighting, staged transfer learning, and fine-tuning. On the held-out image test split, it achieved 96.52% accuracy, 96.50% weighted F1, and 96.95% macro F1. A Grad-CAM module provides visual evidence, while a local OpenCV dashboard performs continuous monitoring with five-prediction temporal averaging, low-confidence warnings, defect-specific causes, corrective actions, and optional OctoPrint telemetry. The 4.06-million-parameter model occupies 41.23 MB and required approximately 108.3 ms per single-image CPU inference on the test laptop. These results support low-cost operator decision assistance, but they do not establish cross-printer generalization because the evaluation uses an image-level split. External printer-session validation and group-aware splitting are therefore identified as necessary next steps.

**Keywords:** additive manufacturing, convolutional neural network, defect classification, EfficientNetB0, FDM, explainable AI, real-time monitoring, transfer learning.

## I. Introduction

Fused deposition modeling, also called fused filament fabrication, constructs a component layer by layer by extruding thermoplastic material. Its low equipment cost has encouraged wide use, but print quality remains sensitive to bed adhesion, extrusion flow, temperature, motion, cooling, and mechanical alignment [1], [2]. Warping, stringing, under-extrusion, layer shifting, cracking, detachment, and spaghetti-like collapse can waste filament and machine time when they are not recognized early.

Human observation is simple but intermittent and subjective. Camera-based monitoring offers a non-contact alternative that can operate without modifying the printer. Earlier studies demonstrated remote stringing detection [3], transfer learning for spaghetti-failure monitoring [4], lightweight object detection for five FFF defects [5], and lightweight CNN classification for FDM defects [6]. These studies establish the feasibility of vision-based monitoring, while also exposing recurring challenges: limited labeled data, class imbalance, device constraints, and sensitivity to camera domain shift.

This work develops a complete operator-facing system rather than stopping at an offline classifier. Its contributions are:

1. An expanded eight-class FDM image-classification dataset and reproducible TensorFlow training pipeline.
2. An EfficientNetB0 transfer-learning model evaluated with accuracy, precision, recall, F1-score, a confusion matrix, and Grad-CAM.
3. A continuous OpenCV dashboard with temporal prediction averaging and explicit uncertainty handling.
4. Defect-specific cause and correction guidance plus optional, clearly separated OctoPrint telemetry.

## II. Related Work

Paraskevoudis et al. applied computer vision and artificial intelligence to remote stringing detection and discussed future intervention in printing parameters [3]. Kim et al. compared transfer-learning and fine-tuning strategies for spaghetti-shaped failure using VGG19, InceptionV3, ResNet50, and EfficientNetB0; their best result reached 94% accuracy and was examined using Grad-CAM [4]. Hu et al. developed a lightweight improved YOLOv8 detector for five FFF defects, reporting 97.5% mAP50 with lower computational cost [5]. Their object-detection formulation localizes defects, whereas the present work predicts a global image condition.

Kuriachen et al. proposed lightweight CNNs for real-time multi-class FDM monitoring and highlighted limitations in generalizability [6]. Kozhay et al. also examined CNN-based FDM defect detection [7]. Aktepe and Ergün compared EfficientNetB0 and MobileNetV2 on the original 1,912-image FDM Kaggle dataset, reporting 87.7% and 97% accuracy, respectively [8]. Because that work closely matches the base data source, the contribution claimed here is not the use of transfer learning itself; it is the expanded eight-class dataset, the retained-model comparison, explainability, uncertainty-aware continuous monitoring, and integrated operator feedback.

EfficientNet uses compound scaling to balance network depth, width, and input resolution [9]. This makes EfficientNetB0 a practical transfer-learning backbone for resource-constrained deployment. Grad-CAM uses gradients flowing into the final convolutional layer to produce a class-discriminative localization map without changing or retraining the network [10].

## III. Materials and Methods

### A. Dataset Preparation

The final dataset contains 4,405 images from compatible public FDM image datasets, including the FDM 3D Printing Defect Dataset [11]. Folder names, text annotations, and CSV labels were normalized into a consistent folder-per-class format. Ambiguous broad labels were excluded from the final specific-class model. The resulting class counts were: cracking 472, layer shifting 364, no defect 900, off platform 91, spaghetti 240, stringing 900, under-extrusion 900, and warping 538. No corrupt image was found during inspection.

Images were divided using seeded stratified sampling: 70% for training and 15% each for validation and testing. This preserves class proportions but does not group frames by recording session. Consequently, the split measures image-level discrimination and may allow correlated frames from one print session to occur in different subsets.

### B. Preprocessing and Augmentation

Every image is decoded as three-channel RGB, resized to 224 x 224 pixels, and represented as float32 values. The training graph applies random horizontal flipping, rotation, zoom, and contrast variation. TensorFlow data pipelines use parallel mapping, batching, and prefetching. Class weights are computed from the training labels to reduce bias toward large classes; the rare off-platform class receives the largest weight.

### C. Network and Optimization

The classifier uses an ImageNet-pretrained EfficientNetB0 feature extractor [9]. The original classification head is removed. Global average pooling, batch normalization, dropout, and an eight-unit softmax layer form the task-specific head. Training proceeds in two stages. First, the backbone is frozen while the classification head learns the FDM categories. Second, the final 30 backbone layers are made trainable and optimized with a smaller learning rate. Adam and sparse categorical cross-entropy are used. Model checkpointing retains the highest validation-accuracy model, while early stopping and learning-rate reduction limit overfitting.

### D. Explainability and Continuous Monitoring

Grad-CAM is computed from the final spatial feature layer to visualize image regions that influence a selected class [10]. For live use, OpenCV reads the local camera and performs inference at a configurable interval. A rolling mean of five probability vectors reduces rapid label changes. Any top confidence below 60% is marked uncertain and accompanied by a warning against changing printer settings from that frame alone.

The dashboard maps a recognized class to a curated description, likely causes, and corrective actions. These recommendations are an engineering knowledge layer, not a generative language model. Optional OctoPrint integration reads printer state, nozzle and bed temperatures, completion percentage, and remaining time. Unavailable speed or axis values remain explicitly unavailable. The software does not automatically move an axis, change a heater, or stop a print.

## IV. Experimental Results

### A. Overall Performance

The retained EfficientNetB0 model achieved a test loss of 0.1058, accuracy of 96.52%, macro F1 of 96.95%, and weighted F1 of 96.50% on 661 held-out images. A separate EfficientNetB3 run achieved slightly higher macro F1 but lower accuracy and weighted F1; therefore, EfficientNetB0 remained the active model.

| Class | Precision / Recall / F1 | n |
|---|---:|---:|
| Cracking | 0.9855 / 0.9577 / 0.9714 | 71 |
| Layer shifting | 0.9815 / 0.9815 / 0.9815 | 54 |
| No defect | 0.9837 / 0.8963 / 0.9380 | 135 |
| Off platform | 0.9333 / 1.0000 / 0.9655 | 14 |
| Spaghetti | 0.9730 / 1.0000 / 0.9863 | 36 |
| Stringing | 0.9783 / 1.0000 / 0.9890 | 135 |
| Under-extrusion | 0.9161 / 0.9704 / 0.9424 | 135 |
| Warping | 0.9756 / 0.9877 / 0.9816 | 81 |

The lowest recall occurred for no-defect images, while the lowest precision occurred for under-extrusion. This pairing is consistent with confusion between subtle flow deficiencies and otherwise acceptable printed regions. Off-platform recall was perfect in the test set, but its support was only 14 images, so this result has high statistical uncertainty.

### B. Deployment Characteristics

The saved model has 4,059,819 parameters and a file size of 41.23 MB. Thirty timed single-image CPU predictions, after five warm-up runs, averaged 108.3 ms on the Ryzen 5 5600H laptop, corresponding to approximately 9.2 serial inferences per second. The dashboard intentionally predicts less frequently than the camera capture loop to keep the interface responsive. Runtime varies with operating-system load and should not be treated as a hardware-independent benchmark.

## V. Discussion

The results show that an EfficientNetB0 transfer-learning model can separate eight visually distinct FDM conditions in the assembled dataset. Class weighting supports minority categories, while temporal averaging makes live output easier to interpret. The system adds operational context absent from a bare softmax classifier: confidence, uncertainty, root-cause prompts, corrective actions, explainability, and optional printer telemetry.

Comparison with prior work requires care because datasets and metrics differ. The 97.5% mAP50 reported by Hu et al. [5] describes object detection, not image classification. The 94% result reported by Kim et al. [4] concerns spaghetti failure, and Aktepe and Ergün [8] use the original five-class dataset. The present 96.52% accuracy therefore should not be interpreted as direct superiority. Its significance lies in eight-class coverage and integrated low-cost deployment.

## VI. Threats to Validity and Limitations

The principal limitation is the stratified image-level split. Images captured close together may share geometry, lighting, printer, and background, allowing session-specific cues to influence the result. A group-aware split by print recording is required for a stronger generalization claim. The merged sources may also contain different label definitions, and the small off-platform and spaghetti classes produce wider uncertainty than their point estimates suggest.

The model performs whole-image classification and does not localize multiple simultaneous defects. Camera views unrelated to FDM printing can still receive a nearest-class score; the low-confidence rule reduces but does not eliminate this open-set problem. Temperature, speed, and axis values cannot be inferred reliably from a single RGB frame. Real telemetry must come from the printer controller. Finally, the current system has not been validated through a long-duration deployment across multiple physical printers, materials, lighting conditions, and camera positions.

## VII. Conclusion

An explainable eight-class FDM defect-monitoring system was developed using EfficientNetB0 transfer learning. The model achieved 96.52% image-level test accuracy and 96.50% weighted F1 on the expanded dataset. The practical system combines continuous OpenCV monitoring, temporal smoothing, uncertainty warnings, Grad-CAM, defect-specific feedback, and optional OctoPrint telemetry. These features make the project suitable as an operator decision-support prototype. The next publication-grade experiment should perform recording-grouped training and external multi-printer validation; closed-loop printer control should be considered only after those tests and explicit hardware safety validation.

## Data and Code Availability

Source code, trained model metadata, evaluation reports, and reproducible scripts are available at https://github.com/Ronnirvin2006/fdm-defect-detection. The public base data source is the FDM 3D Printing Defect Dataset [11]. Additional dataset licenses and redistribution conditions must be checked before republishing merged images.

## Acknowledgment

The author acknowledges the creators of the public FDM image datasets and the maintainers of TensorFlow, Keras, OpenCV, and scikit-learn. Replace this paragraph with project supervisor, department, laboratory, and funding acknowledgments before submission.

## References

[1] T. D. Ngo, A. Kashani, G. Imbalzano, K. T. Q. Nguyen, and D. Hui, “Additive manufacturing (3D printing): A review of materials, methods, applications and challenges,” *Composites Part B: Engineering*, vol. 143, pp. 172–196, 2018, doi: 10.1016/j.compositesb.2018.02.012.

[2] S. Wickramasinghe, T. Do, and P. Tran, “FDM-based 3D printing of polymer and associated composite: A review on mechanical properties, defects and treatments,” *Polymers*, vol. 12, no. 7, art. 1529, 2020, doi: 10.3390/polym12071529.

[3] K. Paraskevoudis, P. Karayannis, and E. P. Koumoulos, “Real-time 3D printing remote defect detection (stringing) with computer vision and artificial intelligence,” *Processes*, vol. 8, no. 11, art. 1464, 2020, doi: 10.3390/pr8111464.

[4] H. Kim, H. Lee, and S.-H. Ahn, “Systematic deep transfer learning method based on a small image dataset for spaghetti-shape defect monitoring of fused deposition modeling,” *Journal of Manufacturing Systems*, vol. 65, pp. 439–451, 2022, doi: 10.1016/j.jmsy.2022.10.009.

[5] W. Hu, C. Chen, S. Su, J. Zhang, and A. Zhu, “Real-time defect detection for FFF 3D printing using lightweight model deployment,” *The International Journal of Advanced Manufacturing Technology*, vol. 134, pp. 4871–4885, 2024, doi: 10.1007/s00170-024-14452-4.

[6] B. Kuriachen, R. Jeyaraj, D. Raphael, P. Ashok, P. S. S. Sundari, and A. Paul, “Defect detection in fused deposition modelling using lightweight convolutional neural networks,” *Engineering Applications of Artificial Intelligence*, vol. 141, art. 109802, 2025, doi: 10.1016/j.engappai.2024.109802.

[7] K. Kozhay, S. Turarbek, T. Asselbekova, M. H. Ali, and E. Shehab, “Convolutional neural network-based defect detection technique in FDM technology,” *Procedia Computer Science*, vol. 231, pp. 119–128, 2024, doi: 10.1016/j.procs.2023.12.183.

[8] E. Aktepe and U. Ergün, “Deep learning application for image-based defect detection in 3D printing processes,” *Journal of Materials and Mechatronics: A*, vol. 7, no. 1, pp. 109–121, 2026, doi: 10.55546/jmm.1910657.

[9] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in *Proc. 36th International Conference on Machine Learning*, vol. 97, 2019, pp. 6105–6114.

[10] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in *Proc. IEEE International Conference on Computer Vision*, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[11] wengmhu, “FDM 3D Printing Defect Dataset,” Kaggle, 2024. [Online]. Available: https://www.kaggle.com/datasets/wengmhu/fdm-3d-printing-defect-dataset. Accessed: Aug. 11, 2026.

[12] H. Wen, C. Huang, and S. Guo, “The application of convolutional neural networks (CNNs) to recognize defects in 3D-printed parts,” *Materials*, vol. 14, no. 10, art. 2575, 2021, doi: 10.3390/ma14102575.
