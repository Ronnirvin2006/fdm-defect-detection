A Real-Time Vision System for Multi-Class Defect Detection in Fused Deposition Modelling Using EfficientNetB0

| Ron Nirvin J; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India; 2023511004 | Ron Nirvin J; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India; 2023511004 | Dharunkumar V; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India; 2023511007 | Dharunkumar V; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India; 2023511007 | Lalith Kishore D; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India; 2023511009 | Lalith Kishore D; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India; 2023511009 |
|---|---|---|---|---|---|
| Dr. Pavithra E; Assistant Professor; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India | Dr. Pavithra E; Assistant Professor; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India | Dr. Pavithra E; Assistant Professor; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India | Dr. Arun Prakash C; Assistant Professor; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India | Dr. Arun Prakash C; Assistant Professor; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India | Dr. Arun Prakash C; Assistant Professor; Department of Production Technology; Madras Institute of Technology, Anna University; Chennai, India |



Abstract—Fused Deposition Modelling (FDM) can continue wasting material after a visible failure develops. This work integrates an eight-condition EfficientNetB0 classifier with a browser-based inspection workflow for remote operators. A historical 4,405-image experiment obtained 96.52% accuracy on a 661-image stratified test split; possible recording overlap limits that result. Subsequent offline continuation used 18,199 images, with 13,772 training and 4,427 recording/date-grouped validation images. The selected checkpoint reached 83.83% accuracy, 83.57% macro F1 and 83.75% weighted F1 on this reused model-selection validation set, not an independent test. A direct HTTPS interface keeps the camera preview local, analyses the newest frame without a server backlog, and presents stable decisions, uncertainty, curated causes and corrective checks. Photo upload, video playback analysis, snapshots, CSV export and on-demand Grad-CAM are integrated into a responsive light/dark dashboard. Face and image-quality checks plus operator view confirmation reduce accidental non-printer decisions but do not establish open-set recognition. Browser verification uses simulated media; external printer trials, calibrated alerts and machine control remain outside the demonstrated evidence.

Keywords—additive manufacturing, convolutional neural network, defect classification, EfficientNetB0, FDM, explainable AI, real-time monitoring, transfer learning.

I. INTRODUCTION

Fused Deposition Modelling, also called fused filament fabrication, constructs a component layer by layer by depositing heated thermoplastic through a moving nozzle. The method is common in laboratories, education, prototyping, and small-scale manufacturing because the equipment and material are comparatively inexpensive. That accessibility does not remove process sensitivity. Bed adhesion, extrusion flow, nozzle and bed temperature, cooling, travel motion, belt tension, and mechanical alignment all influence the printed result [1], [2]. A small disturbance can develop into warping, stringing, under-extrusion, layer shifting, cracking, detachment, or a spaghetti-like collapse.

In a normal laboratory, one person may supervise several machines. Direct observation is useful, but it is intermittent and depends on experience. A camera provides a non-contact view without changing the printer mechanism. A convolutional neural network (CNN) can learn visual patterns that are difficult to describe through fixed thresholds, while transfer learning makes this possible with a moderate dataset. Earlier work demonstrated remote stringing detection [3], transfer learning for spaghetti-failure monitoring [4], lightweight object detection for five FFF defects [5], and lightweight CNN classification for FDM defects [6]. These studies establish that vision-based monitoring is feasible, but they also expose recurring problems: limited labelled data, unequal class sizes, restricted computing hardware, changes in camera viewpoint, and models that report a class without helping the operator decide what to inspect.

The system connects camera acquisition, classification and operator interpretation. A remote phone supplies live images to a host that runs the model. Class scores remain separate from physical process measurements: the application does not invent temperature, motion or nozzle values from an RGB image. The final interface reports recommendations only; printer control is not part of this submission.

The contribution is an application-level integration and evaluation, not a new neural architecture, transfer-learning algorithm, or explanation method. EfficientNetB0 and Grad-CAM are established components [9], [10]. The specific contributions are:

A documented eight-condition classification task assembled from two public FDM sources, with harmonised labels and exclusion of ambiguous generic defect labels.

An auditable separation of historical image-level testing from later recording/date-grouped continuation, with retained split paths, probabilities and class-wise errors.

A remote browser workflow with latest-frame processing, view rejection, consistent-reading alerts, Grad-CAM, troubleshooting guidance and downloadable inspection records.

The merged collection is a task-specific reuse of public data rather than a newly captured dataset. The study does not establish algorithmic superiority or claim that temporal averaging improves classification accuracy without a sequence-level experiment.

II. RELATED WORK

Paraskevoudis et al. applied computer vision and artificial intelligence to remote stringing detection and discussed future intervention in printing parameters [3]. Kim et al. compared transfer-learning and fine-tuning strategies for spaghetti-shaped failure using VGG19, InceptionV3, ResNet50, and EfficientNetB0; their ResNet50 deployment exceeded 90% accuracy and was examined using Grad-CAM [4]. Hu et al. developed a lightweight improved YOLOv8 detector for five FFF defects, reporting 97.5% mAP50 with lower computational cost [5]. Their object-detection formulation localises defects, whereas the present work predicts a global image condition.

Kuriachen et al. proposed lightweight CNNs for real-time multi-class FDM monitoring and highlighted limitations in generalisability [6]. Kozhay et al. also examined CNN-based FDM defect detection [7]. Aktepe and Ergün compared EfficientNetB0 and MobileNetV2 on the original 1,912-image FDM Kaggle dataset, reporting 87.7% and 97% accuracy, respectively [8]. Because that work closely matches the base data source, the contribution claimed here is not the use of transfer learning itself; it is the expanded eight-class dataset, detailed retained-model evaluation, explainability, uncertainty-aware continuous monitoring, and integrated operator feedback.

EfficientNet uses compound scaling to balance network depth, width, and input resolution [9]. This makes EfficientNetB0 a practical transfer-learning backbone for resource-constrained deployment. Grad-CAM uses gradients flowing into the final convolutional layer to produce a class-discriminative localisation map without changing or retraining the network [10].

The literature also shows why reported percentages cannot be compared in isolation. Object-detection mAP, binary failure accuracy, and multi-class image accuracy answer different questions. Dataset size, class definitions, train-test separation, and camera conditions further change task difficulty. The present study therefore reports per-class precision, recall, F1, and support for the retained model alongside overall accuracy.

III. MATERIALS AND METHODS

A. Dataset Preparation

Two public sources define the eight-condition task [11], [12]. The first supplies cracking, layer shifting, off-platform, stringing and warping images; the second supplies good, under-extrusion, stringing and spaghetti frames with recording metadata. Labels are harmonised to one class order; Good becomes No defect, and generic Defected labels are excluded to avoid overlapping categories. The historical experiment used a capped 4,405-image collection (Table I). The restored continuation collection contains 18,199 images: 472 cracking, 364 layer shifting, 7,026 no defect, 91 off platform, 240 spaghetti, 3,956 stringing, 5,512 under-extrusion and 538 warping images. No corrupt image was found in the prepared collection.

TABLE I
HISTORICAL STRATIFIED IMAGE-LEVEL SPLIT

| Class | Total | Train | Validation | Test |
|---|---|---|---|---|
| Cracking | 472 | 330 | 71 | 71 |
| Layer shifting | 364 | 255 | 55 | 54 |
| No defect | 900 | 630 | 135 | 135 |
| Off platform | 91 | 64 | 13 | 14 |
| Spaghetti | 240 | 168 | 36 | 36 |
| Stringing | 900 | 630 | 135 | 135 |
| Under-extrusion | 900 | 630 | 135 | 135 |
| Warping | 538 | 376 | 81 | 81 |
| Total | 4,405 | 3,083 | 661 | 661 |


The historical 70/15/15 stratified split used seed 42, producing 3,083/661/661 images. Its original split manifest is unavailable and recording overlap cannot be excluded. Continuation instead preserves Nimbus recording identifiers and groups original FDM filenames by capture date, keeping related copies together. The fixed split contains 13,772 training and 4,427 validation images. Grouping reduces within-recording leakage in this split but cannot remove unknown overlap with the source checkpoint's earlier training.

B. Preprocessing and Augmentation

Every image is decoded as three-channel RGB, resized to 224 x 224 pixels, and represented as float32 values. Training-only augmentation applies random horizontal flipping, rotation up to approximately 5% of a full turn, zoom up to 8%, and contrast variation up to 8%. Validation and test images are resized but not randomly augmented. TensorFlow data pipelines use parallel mapping, batching, and prefetching.

Class imbalance is substantial, especially the 91 off-platform images. Historical training used balanced weights, computed as

w_c = N / (K n_c),

where N is training size, K the class count and n_c the class support. Local continuation used square-root balanced weights; the final experiment additionally multiplied cracking, no-defect, spaghetti and under-extrusion weights by 1.35. Weighting changes loss contributions without creating independent minority examples.

C. Network and Optimisation

The classifier uses an ImageNet-pretrained EfficientNetB0 feature extractor [9]. Its original ImageNet classification head is removed. Global average pooling converts the spatial feature maps to a compact vector, after which batch normalisation, dropout (0.25), and an eight-unit softmax layer form the task-specific head. For class c, softmax converts logit z_c into probability

p_c = exp(z_c) / sum_j exp(z_j).

Weighted sparse categorical cross-entropy is -w_y log(p_y). Historical training used Adam, batch size 32, initial head/backbone learning rates 0.001/0.00001, and the final 30 backbone layers for fine-tuning (Table II). Offline continuation on a GTX 1650 used batch size four and loss-based checkpoint selection. The final weak-class run completed two head epochs at 0.00001 and three backbone epochs at 0.000001 before early stopping. Batch-normalisation backbone layers remained frozen. Each continuation loaded prior weights with a new optimiser; this is not an exact interrupted-optimiser resume.

TABLE II
HISTORICAL NETWORK AND TRAINING CONFIGURATION

| Setting | Retained value |
|---|---|
| Input resolution | 224 x 224 x 3 |
| Backbone | EfficientNetB0, ImageNet weights |
| Classification head | GAP, BatchNorm, Dropout, Dense |
| Output classes | 8 |
| Batch size | 32 |
| Initial / fine-tuning learning rate | 0.001 / 0.00001 |
| Frozen / fine-tuning schedule | 25 / 10 epoch limits |
| Fine-tuned backbone layers | Final 30 |
| Checkpoint criterion | Highest validation accuracy |


D. Evaluation Measures

Accuracy gives the fraction of all correctly classified test images. For each class, precision measures how many images predicted as that class are correct, recall measures how many true images of that class are recovered, and F1 is their harmonic mean:

Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2PR / (P + R).

Macro averaging gives each class equal importance, whereas weighted F1 reflects support. A Wilson 95% interval accompanies the historical image-level accuracy; it is not a cluster-aware confidence interval. Current grouped validation is reused for checkpoint selection and must not be presented as a new untouched test.

E. Explainability and Continuous Monitoring

Grad-CAM is computed from the final spatial feature layer to visualise image regions that influence a selected class [10]. It weights each feature map using the spatially averaged gradient of the selected output, combines the weighted maps, applies a rectified linear unit, and overlays the result on the input image. The heat map is evidence of model attention, not a segmented defect boundary.

The browser preview runs independently of inference. Each completed response schedules a fresh JPEG frame after a 35 ms scheduling floor; only one request per browser is in flight. Older camera frames are discarded rather than queued. Consequently, decision rate depends on host and network latency, not preview FPS. A 70% softmax-score threshold and three consecutive agreeing predictions produce the displayed confirmed condition. Other valid frames are labelled checking or uncertain; rejected views reset agreement history.

Confirmation: c_t = c_(t-1) = c_(t-2), with min(s_t, s_(t-1), s_(t-2)) >= 0.70.

The earlier local prototype averaged five prediction vectors at a nominal 0.8 s interval, spanning about 3.2 s from oldest to newest. That window was a presentation heuristic, not a result of printer-speed optimisation. The final browser version replaces averaging with agreement checks to avoid a fixed multi-second delay. Neither rule has a physical transient-failure or window-length ablation. Three agreeing updates need two inter-update intervals after the first qualifying result; a brief event may be missed. This is an analytical property, not a measured alarm guarantee.

Before camera analysis, the operator confirms that the print is visible. Frontal/profile face checks and extreme lighting/near-uniform-view rejection run before classification. These are preliminary heuristics, not trained nozzle or bed detectors. A score threshold cannot guarantee rejection of arbitrary objects; all confirmed alerts still require visual review.

Grad-CAM is available on demand for the current frame, photo or snapshot rather than every live update. This keeps explanation cost separate from continuous inference. The overlay visualises model attention and does not establish a defect boundary or causal explanation.

The light/dark interface places confidence, possible causes and recommended checks beside the live view. Guidance is a fixed, class-conditioned engineering knowledge layer, not unrestricted generated advice. Photo and snapshot analysis use one unsmoothed prediction; video playback is inspected continuously at the inference rate, so not every encoded video frame is evaluated. CSV records include timestamp, source, video time, class score, displayed state, processing time and round-trip time. Camera tracks are released on Stop, mode changes, page hiding and page exit. No automatic heater, axis, cleaning or pause command is issued.

TABLE III
OPERATOR GUIDANCE IMPLEMENTED IN THE DASHBOARD

| Predicted class | Visual interpretation | First operator check |
|---|---|---|
| Cracking | Split or cracked printed region | Temperature, cooling, and enclosure |
| Layer shifting | Layers displaced in X or Y | Belts, pulleys, speed, and obstruction |
| No defect | No trained defect is dominant | Confidence and continued observation |
| Off platform | Part detached or outside expected bed area | Bed level, Z-offset, and adhesion |
| Spaghetti | Loose tangled filament after print failure | Stop safely, then inspect adhesion and supports |
| Stringing | Thin strands between separated regions | Retraction, nozzle temperature, and filament moisture |
| Under-extrusion | Gaps, sparse lines, or insufficient flow | Nozzle blockage, extruder grip, and flow calibration |
| Warping | Corners or edges curling upward | Bed temperature, first layer, and cooling |


Table III summarises only the first diagnostic direction. The operator must confirm the physical condition before changing a setting, follow the printer and filament limits, and stop or cool the machine safely before touching the hot end.

IV. EXPERIMENTAL RESULTS


Fig. 1. Confusion matrix for the 661-image held-out test split.

A. Overall Performance

The historical checkpoint correctly classified 638 of 661 held-out images: 96.52% accuracy, test loss 0.1058, macro F1 96.95% and weighted F1 96.50%. Its Wilson 95% image-level interval was 94.83-97.67%. Table IV and Figures 1-2 describe this historical experiment only, not the newly deployed continuation checkpoint.

TABLE IV
PER-CLASS PERFORMANCE ON THE HELD-OUT TEST SPLIT

| Class | Precision / Recall / F1 | n |
|---|---|---|
| Cracking | 0.9855 / 0.9577 / 0.9714 | 71 |
| Layer shifting | 0.9815 / 0.9815 / 0.9815 | 54 |
| No defect | 0.9837 / 0.8963 / 0.9380 | 135 |
| Off platform | 0.9333 / 1.0000 / 0.9655 | 14 |
| Spaghetti | 0.9730 / 1.0000 / 0.9863 | 36 |
| Stringing | 0.9783 / 1.0000 / 0.9890 | 135 |
| Under-extrusion | 0.9161 / 0.9704 / 0.9424 | 135 |
| Warping | 0.9756 / 0.9877 / 0.9816 | 81 |


The lowest recall occurred for no-defect images, while the lowest precision occurred for under-extrusion. This pairing is consistent with confusion between subtle flow deficiencies and otherwise acceptable printed regions. Off-platform recall was perfect in the test set, but its support was only 14 images, so this result has high statistical uncertainty.

B. Class-Wise Interpretation

Historical stringing and spaghetti F1 scores were 98.90% and 98.63%. However, no-defect recall was 89.63%, with 12 acceptable images assigned to under-extrusion. This healthy/low-flow boundary is more consequential operationally than the aggregate score suggests.

The 23 historical errors include three cracking, one layer-shifting, fourteen no-defect, four under-extrusion and one warping example. Correct recovery of every off-platform image involves only fourteen examples, giving a wide 78.47-100% Wilson recall interval. Such minority support does not establish dependable field performance.

C. Deployment Characteristics

The historical model has 4,059,819 parameters and occupies 41.23 MB including its saved training state. Its thirty single-image CPU model calls, after five warm-ups, averaged 108.3 ms (median 107.6 ms) on the Ryzen 5 5600H laptop. These timings exclude capture, transport and alert confirmation. The selected continuation keeps the architecture but uses a different saved checkpoint; the historical timing is not assigned to the new network runtime.

D. Interface Behaviour and Scope of Evidence

Automated browser checks exercise continuous acquisition, view confirmation, light/dark rendering, photo/video input, snapshots, Grad-CAM, CSV output and camera release. They use a synthetic camera and responsive browser viewports, not Android/iOS hardware or a physical printer. Network tests traverse a public HTTPS tunnel. These checks establish software-path operation, not physical defect sensitivity or cross-printer reliability. The reference machine-integration code is not enabled in the final public workflow. Local simulated-camera checks completed 11.43 updates/s in a desktop viewport and 11.85 updates/s in a mobile viewport. These short loopback checks are not physical camera, network-wide or failure-detection benchmarks. In corresponding short public-HTTPS simulated-camera runs, measured rates were 2.86-3.57 updates/s. These rates depend on the observed network and do not guarantee another device's throughput.

E. Comparison with Published FDM Systems

Table V compares reported task scope and metrics from the cited studies. Values are transcribed from their publisher or author-institution records, rather than reproduced on the present split. NR denotes a value not established in those consulted records.

TABLE V
COMPARISON WITH PUBLISHED FDM SYSTEMS

| Study / model | Task and data | Reported metric | Timing / scope |
|---|---|---|---|
| Kim et al. [4], ResNet50 | Spaghetti failure | Accuracy >90% | About 1 s; small device |
| Hu et al. [5], improved YOLOv8 | Five-defect detection | mAP50 97.5% | FPS +18.1% vs own baseline |
| Aktepe and Ergun [8], B0 | Five-class; 1,912 images | Accuracy 87.7% | Timing NR |
| Aktepe and Ergun [8], MobileNetV2 | Same source study | Accuracy 97.0% | Timing NR |
| Present EfficientNetB0 | Eight-class; 661 test images | Accuracy 96.52%; macro F1 96.95% | 108.3 ms; CPU model call |


The table provides context, not a controlled ranking: detection mAP50 cannot be equated with classification accuracy, and hardware, source data, class count, and splits differ. A same-split MobileNetV2/ResNet50 baseline and smoothing ablation have not been retained, so neither backbone superiority nor an accuracy gain from the interface can be concluded. The practical distinction is eight-condition coverage linked to class-wise error analysis and operator guidance.

F. Exploratory Grouped Continuation Experiment

On the fixed 4,427-image grouped validation set, the historical checkpoint obtained 51.84% accuracy and 57.62% macro F1. Full-data continuation reached 83.10% accuracy and 84.98% macro F1; a short follow-up reached 83.49% and 83.42%. The final weak-class checkpoint achieved 83.83% accuracy, 83.57% macro F1 and 83.75% weighted F1, with loss 0.5254. It improved accuracy by fifteen images over the short run but did not improve every class. Spaghetti recall remains 58.33% (28/48); no-defect recall is 73.94%. This final checkpoint is selected for the demonstration, while all earlier weights are preserved. Validation was reused for model selection and source-checkpoint overlap is unknown: none of these figures is an independent test or directly comparable with the historical 96.52% result.

TABLE VI
SELECTED CHECKPOINT: GROUPED VALIDATION

| Class | Precision / Recall / F1 | n |
|---|---|---|
| Cracking | 0.7188 / 0.8679 / 0.7863 | 106 |
| Layer shifting | 1.0000 / 0.7536 / 0.8595 | 138 |
| No defect | 0.9329 / 0.7394 / 0.8250 | 1938 |
| Off platform | 1.0000 / 1.0000 / 1.0000 | 22 |
| Spaghetti | 0.8000 / 0.5833 / 0.6747 | 48 |
| Stringing | 0.8314 / 0.9868 / 0.9024 | 1214 |
| Under extrusion | 0.6923 / 0.8680 / 0.7703 | 788 |
| Warping | 0.8671 / 0.8671 / 0.8671 | 173 |

V. DISCUSSION


Fig. 2. Class-wise test performance. Parentheses show test support; Under-extr. denotes under-extrusion.

A. Error Patterns and Operator Decisions

Historical and current evaluations expose the same practical boundary: healthy surface variation can resemble weak extrusion. In grouped validation, spaghetti recall of 58.33% is a stronger warning than its 67.47% F1 alone. Three-update agreement may reduce visible flicker, but no alarm-accuracy improvement is claimed without labelled sequences. Curated advice suggests checks; it does not identify a verified physical root cause.

B. Calibration and Uncertainty Thresholds

Guo et al. [13] motivate post-hoc temperature scaling: a positive scalar fitted on calibration logits adjusts confidence without changing the top class. Expected calibration error (ECE) instead measures the absolute gap between confidence and empirical correctness across bins:

For M confidence bins, expected calibration error (ECE) compares the mean top-class confidence and empirical accuracy within each bin:

ECE = Σ_b (n_b / n) |a_b − c_b|, b = 1, ..., M.

For the selected checkpoint, fifteen equal-width bins on the reused grouped validation probabilities give ECE 0.0635, negative log-likelihood 0.5254 and multiclass Brier score 0.2565. Bin counts and probabilities are retained. These are descriptive, uncalibrated model-selection diagnostics, not independent calibration evidence. Historical paired probabilities are unavailable.


Fig. 3. Raw reliability on reused grouped validation. The dashed diagonal denotes perfect calibration; no scaling was fitted.

The earlier 60% warning and current 70% confirmation thresholds are interface heuristics, not statistically justified safety cutoffs. No fitted temperature, threshold optimum or before/after calibration gain is claimed. Calibration needs a separate subset and final recording-grouped or external testing; temporal decision rules also require sequence-level assessment.

VI. THREATS TO VALIDITY AND LIMITATIONS

The historical image split may share recording-specific cues across subsets. Later grouping by Nimbus recording and FDM capture date is stronger, but its validation set was repeatedly inspected and its relation to earlier checkpoint training is unknown. Therefore the present record does not establish independent cross-printer generalisation. Small off-platform and spaghetti supports and mixed source definitions further limit precision.

Whole-image classification cannot localise simultaneous defects. Face and lighting gates plus explicit operator confirmation reduce accidental background analysis, but arbitrary out-of-domain scenes can still pass. The confidence scores are uncalibrated. Synthetic browser checks are not long-duration physical trials, do not verify actual Android/iOS camera hardware and do not establish false-alarm or time-to-detection performance. Neither temperatures nor motion can be recovered reliably from a camera score.

A follow-up should reserve an external printer-camera setup, record healthy and faulty sequences with timestamps, and measure sensitivity, false alarms and decision delay. Printer actuation must remain outside automatic control until confidence calibration and hardware safety checks are completed. The software can support this experiment, but missing physical evidence cannot be replaced by additional training epochs.

VII. CONCLUSION

An eight-condition FDM inspection system links EfficientNetB0 with continuous browser-camera analysis, images, videos, snapshots, on-demand Grad-CAM and operator feedback. Historical image-level testing yielded 96.52% accuracy; later grouped model-selection validation yielded 83.83% for the selected demonstration checkpoint. Keeping these results separate prevents an unsupported field-accuracy claim. The final application preserves a responsive preview while processing the latest frame without a backlog and exports auditable CSV records. Its contribution is an integrated decision-support workflow, not a new architecture or validated autonomous printer controller. External physical-printer and calibration evaluation remain necessary.

DATA AND CODE AVAILABILITY

Source code, trained model metadata, evaluation reports, and reproducible scripts are available at https://github.com/Ronnirvin2006/fdm-defect-detection. The public training sources are cited in [11] and [12]. Their licenses and redistribution conditions must be checked before republishing merged images.

REPRODUCIBILITY NOTES

The historical model, aggregate reports and project-derived figures are preserved separately from the selected continuation checkpoint, which retains split paths and per-image probabilities. Browser checks retain source, logs and screenshots. Source data remain subject to their original licences. No physical printer result, platform-wide similarity percentage or calibrated safety claim is asserted.

ACKNOWLEDGMENT

The authors thank the Department of Production Technology, Madras Institute of Technology, Anna University, for supporting this project. They also acknowledge the creators of the public FDM image datasets and the maintainers of TensorFlow, Keras, OpenCV, and scikit-learn.

REFERENCES

[1] T. D. Ngo, A. Kashani, G. Imbalzano, K. T. Q. Nguyen, and D. Hui, “Additive manufacturing (3D printing): A review of materials, methods, applications and challenges,” Composites Part B: Engineering, vol. 143, pp. 172–196, 2018, doi: 10.1016/j.compositesb.2018.02.012.

[2] S. Wickramasinghe, T. Do, and P. Tran, “FDM-based 3D printing of polymer and associated composite: A review on mechanical properties, defects and treatments,” Polymers, vol. 12, no. 7, art. 1529, 2020, doi: 10.3390/polym12071529.

[3] K. Paraskevoudis, P. Karayannis, and E. P. Koumoulos, “Real-time 3D printing remote defect detection (stringing) with computer vision and artificial intelligence,” Processes, vol. 8, no. 11, art. 1464, 2020, doi: 10.3390/pr8111464.

[4] H. Kim, H. Lee, and S.-H. Ahn, “Systematic deep transfer learning method based on a small image dataset for spaghetti-shape defect monitoring of fused deposition modeling,” Journal of Manufacturing Systems, vol. 65, pp. 439–451, 2022, doi: 10.1016/j.jmsy.2022.10.009.

[5] W. Hu, C. Chen, S. Su, J. Zhang, and A. Zhu, “Real-time defect detection for FFF 3D printing using lightweight model deployment,” The International Journal of Advanced Manufacturing Technology, vol. 134, pp. 4871–4885, 2024, doi: 10.1007/s00170-024-14452-4.

[6] B. Kuriachen, R. Jeyaraj, D. Raphael, P. Ashok, P. S. Sundari, and A. Paul, “Defect detection in fused deposition modelling using lightweight convolutional neural networks,” Engineering Applications of Artificial Intelligence, vol. 141, art. 109802, 2025, doi: 10.1016/j.engappai.2024.109802.

[7] K. Kozhay, S. Turarbek, T. Asselbekova, M. H. Ali, and E. Shehab, “Convolutional neural network-based defect detection technique in FDM technology,” Procedia Computer Science, vol. 231, pp. 119–128, 2024, doi: 10.1016/j.procs.2023.12.183.

[8] E. Aktepe and U. Ergün, “Deep learning application for image-based defect detection in 3D printing processes,” Journal of Materials and Mechatronics: A, vol. 7, no. 1, pp. 109–121, 2026, doi: 10.55546/jmm.1910657.

[9] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. 36th International Conference on Machine Learning, vol. 97, 2019, pp. 6105–6114.

[10] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE International Conference on Computer Vision, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[11] wengmhu, “FDM 3D Printing Defect Dataset,” Kaggle, 2024. [Online]. Available: https://www.kaggle.com/datasets/wengmhu/fdm-3d-printing-defect-dataset. Accessed: Aug. 11, 2026.

[12] nimbus200, “3D Printing Errors,” Kaggle, 2024. [Online]. Available: https://www.kaggle.com/datasets/nimbus200/3d-printing-errors. Accessed: Aug. 11, 2026.

[13] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. 34th International Conference on Machine Learning, vol. 70, 2017, pp. 1321–1330. [Online]. Available: https://proceedings.mlr.press/v70/guo17a.html.
