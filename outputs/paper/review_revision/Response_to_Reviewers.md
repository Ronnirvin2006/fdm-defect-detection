# Response to Reviewers - Paper 54

Title: A Real-Time Vision System for Multi-Class Defect Detection in Fused Deposition Modelling Using EfficientNetB0

The authors thank the reviewers for their constructive comments. The revised manuscript retains the measured eight-class results and distinguishes implemented features, analytical observations, and experiments that remain to be performed.

## Reviewer 1

### 1. Clarify novelty beyond EfficientNetB0, transfer learning and Grad-CAM

Section I now explicitly identifies the contribution as application-level integration and evaluation. No new architecture, transfer-learning algorithm, explanation method, or newly captured dataset is claimed. The contribution is the harmonised eight-condition task, auditable evaluation of the retained model, and a common upload/live workflow with guidance and controller-sourced telemetry. Section III-E clarifies that Grad-CAM is an optional offline diagnostic rather than a per-frame live feature.

### 2. Add a comparison table with existing methods and key metrics

Section IV-E and Table V compare Kim et al., Hu et al., and both models reported by Aktepe and Ergun with the present system. Task scope, reported metric, and available timing information are included. The values were checked against publisher/author-institution records. The table explicitly distinguishes mAP50 from classification accuracy, and different datasets and hardware prevent a controlled ranking. No unperformed baseline experiment is presented.

### 3. Shorten discussion and strengthen experimental comparison

The repeated general discussion and long reproducibility narrative have been shortened. The recovered space is used for Table V, explicit error interpretation, and the calibration discussion. The original 661-image test results remain unchanged. The revision identifies missing same-split backbone comparisons and temporal-window ablations instead of implying they were conducted.

## Reviewer 2

### 1. Improve Fig. 2 labels, legends and vector clarity

The previous wide raster training plot has been replaced by a class-performance plot generated from the retained classification report. It shows precision, recall and F1, with test support in each class label and a percentage axis. Both revised figures are supplied as vector PDF/SVG plus 600 dpi PNG fallbacks for Word. The final PDF embeds the vector plots. The original training image is retained in the project, but no numerical training history was invented to redraw it.

### 2. Explain the five-frame rolling window and transient failures

Section III-E corrects the description to five prediction vectors. The code schedules inference at a default 0.8 s interval, independent of camera FPS. A full window therefore spans approximately 3.2 s between oldest and newest entries. The choice is explicitly a presentation heuristic rather than an optimum derived from print speed. Analytical step-response behaviour and suppression of one-update transients are described; measured physical detection latency and an immediate-event bypass are not claimed.

### 3. Discuss ECE, Platt/temperature scaling and the 60% cutoff

Section V-B adds the ECE definition, a temperature-scaling procedure with validation-only fitting, and a proposed evaluation using reliability diagrams, negative log-likelihood, Brier score, coverage and accepted-prediction accuracy. Guo et al. is cited as [13]. Calibration of the averaged live output is distinguished from calibration of individual frames. The 60% cutoff is explicitly an unvalidated interface heuristic. The retained files lack paired validation/test prediction arrays and the original split manifest, so no ECE, fitted temperature, or statistically justified cutoff is fabricated. This comment is addressed through the requested discussion; empirical calibration remains outstanding.

### 4. Use consistent English

British English is used in the revised prose, including modelling, generalisation, normalisation, optimisation and localisation. Exact titles of published references retain their original spelling.

## Evidence and remaining experiments

The reported accuracy remains 638/661 = 96.52%; weighted F1 is 96.50% and macro F1 is 96.95%. Figure 1 is reconstructed from the retained confusion counts, which sum to 661 and contain 23 errors. Figure 2 uses the retained four-decimal class report. No retraining, new data collection, physical latency experiment, same-split backbone benchmark, calibration fit, or smoothing ablation was performed for this editorial revision.

Checked sources:
- Kim et al.: https://pure.konkuk.ac.kr/en/publications/systematic-deep-transfer-learning-method-based-on-a-small-image-d/
- Hu et al.: https://link.springer.com/article/10.1007/s00170-024-14452-4
- Aktepe and Ergun: https://dergipark.org.tr/en/pub/jmm/article/1910657
- Guo et al.: https://proceedings.mlr.press/v70/guo17a.html
