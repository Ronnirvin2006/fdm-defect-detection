Response to Reviewers

IAME 2026 | Paper 54 | Final revision | 29 October 2026

A Real-Time Vision System for Multi-Class Defect Detection in Fused Deposition Modelling Using EfficientNetB0

Reviewer 1.1 — Novel contribution

The contribution is stated as application-level integration and evaluation, not a new EfficientNet, transfer-learning or Grad-CAM method. The manuscript now separates the historical experiment, grouped continuation and browser workflow, including limits of their evidence.

Reviewer 1.2 — Comparison with existing methods

Table V compares cited FDM systems by task, reported metric and timing scope. Different datasets, class counts and mAP versus accuracy are explicitly not ranked. Section IV-F and Table VI add the completed 4,427-image grouped continuation results: the historical model's 51.84% accuracy on this split versus 83.83% for the selected checkpoint. Reused model-selection validation is not described as an independent test.

Reviewer 1.3 — Shorter discussion, stronger experiments

The repetitive class discussion is shortened. The saved grouped split, per-class results, continuation history and calibration diagnostics now occupy the recovered space. The original 96.52% historical score is retained but is not used as the new deployment score.

Reviewer 2.1 — Figure clarity

All three project-derived manuscript figures are embedded as vector PDF content. Class labels and legends remain searchable. Figures 1-2 are labelled as historical results; Table VI and the new reliability diagram in Figure 3 report the grouped validation. No online photographs or third-party plots are included.

Reviewer 2.2 — Five-frame window and transients

Section III-E explains that the earlier five-prediction window was heuristic, not selected from printer speed or camera FPS. Its nominal span was 3.2 s at 0.8 s per update. The final browser workflow instead processes the newest frame after each response and requires three agreeing predictions at 70% confidence. Short events can still be missed, and no physical failure latency or temporal ablation is claimed.

Reviewer 2.3 — Calibration and the uncertainty threshold

Section V-B discusses temperature scaling and reports descriptive raw calibration diagnostics on the saved grouped validation probabilities: 15-bin ECE 0.0635, NLL 0.5254, multiclass Brier 0.2565. Validation reuse and unknown source-training overlap are stated. No independent calibration assessment, fitted temperature or optimised threshold is claimed. Historical paired probabilities remain unavailable.

Reviewer 2.4 — Consistent English

British spelling is used in the narrative: modelling, normalisation, optimisation, generalisation and labelled. Published reference titles retain their original spelling. Author names use trailing initials; Assistant Professor appears on a separate, non-bold line for the two faculty authors.

Evidence and submission boundary

Browser verification uses simulated media and responsive Chromium viewports, not a physical printer or actual Android/iOS hardware. Printer actuation is not enabled. The acceptance email specifies camera-ready completion by 31 October 2026 and similarity below 15%; the author's stricter under-10% target still requires an institution-approved Turnitin/iThenticate check. No such report is fabricated or included. Supervisor approval, registration and CMT upload remain author actions.
