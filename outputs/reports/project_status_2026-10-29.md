# Project Status - 29 October 2026

## Scope

Inspected `/home/ron/ml`, the FDM repository, its saved artefacts, and its configured GitHub remote. All files created during this check are inside `/home/ron/ml`. No commit, push, cleanup, retraining, or application source change was performed.

## GitHub and Local Changes

- Repository: https://github.com/Ronnirvin2006/fdm-defect-detection
- Branch: `main`.
- `git fetch origin` succeeded. Local HEAD and fetched `origin/main` are both `60eeaac` (`Finalize paper artifacts and dashboard updates`), with zero commits ahead or behind.
- The application source, requirements, active model, and class-name mapping match the fetched GitHub version.
- The working tree contains changes that are not committed or on GitHub:
  - Modified `tools/create_conference_paper.py`: optional table column breaks and a literature-comparison caption.
  - New `tools/revise_camera_ready.py` and `tools/finalise_review_revision.py`.
  - New `outputs/paper/review_revision/`: revised paper, reviewer response, vector figures, and previews.
  - This dated status report.

Equal HEAD commits do not mean that these uncommitted files have been backed up to GitHub.

## Verified Today

- Parsed all 18 Python source/tool files without syntax errors.
- `pip check` in the project virtual environment found no broken installed requirements.
- The Keras model archive passed its ZIP integrity check.
- Loaded the actual active model successfully: input `(None, 224, 224, 3)`, output `(None, 8)`.
- Verified eight labels matching the number of model outputs.
- Started a temporary local HTTP server with the actual predictor and dashboard handlers.
- `GET /` and `GET /api/status` returned HTTP 200.
- `POST /api/upload-image` returned HTTP 200 and eight finite class scores for a synthetic PNG.
- Shut down the test server. No camera was opened. No dashboard was running on the normal 8765 or Streamlit 8501 ports when inspected.
- Revised paper: six PDF pages; reviewer response: two PDF pages. Both editable DOCX archives passed integrity checks.

The synthetic upload validates software plumbing only. Its predicted class is not an accuracy measurement. The rendered browser, physical webcam, camera-release behaviour, and real OctoPrint connection were not tested today.

## Active Model and Saved Results

The application uses `models/best_model.keras` and `models/class_names.json`, not the paper folder or an alternative expanded-results folder. There is one active model location.

Classes: Cracking, Layer shifting, No defect, Off platform, Spaghetti, Stringing, Under-extrusion, and Warping.

Saved experiment: 4,405 images; 3,083 training, 661 validation, and 661 test images. Saved accuracy is 96.52%, weighted F1 is 96.50%, and macro F1 is 96.95%. These are retained historical image-level results, not newly measured scores. The earlier CPU benchmark is approximately 108.3 ms per synthetic single-image model call and excludes full camera/transport/decision latency.

The raw and processed data folders currently contain only placeholders. The trained model is usable without training images, but training or full dataset evaluation requires obtaining and preparing the datasets again. The original selected-image manifest, per-image validation/test probabilities, and numerical training history are absent.

## Current Application Behaviour

- Recommended local dashboard: `src/live_dashboard.py`, default URL `http://127.0.0.1:8765`.
- Supports image upload and continuous OpenCV camera monitoring.
- Live inference defaults to one prediction every 0.8 seconds and averages the last five prediction vectors.
- Uploaded images use an unsmoothed single prediction.
- A score below 60% triggers an uncertainty warning; the threshold is heuristic and uncalibrated.
- Feedback is curated defect knowledge, causes, and corrective checks. It is not an integrated generative AI service.
- OctoPrint telemetry requires a configured printer URL and API key. Temperature and progress are controller measurements; unavailable speed/axis values are not estimated from the image.
- No automatic heater changes, motion commands, or nozzle cleaning are implemented.
- Grad-CAM is an optional separate diagnostic script, not a per-frame live overlay.

## Remaining Work and Documentation Issues

1. Back up the reviewer revision and its generator changes with a commit and push when requested. The application/model currently on GitHub are current, but the revised paper is local only.
2. The README lists obsolete paper paths (`FDM_Defect_Detection_Conference_Paper.docx/.pdf`). The older retained document is named `IAME2026_FDM_Defect_Detection_Full_Paper`, and the latest paper is under `outputs/paper/review_revision/IAME2026_Revised_Paper`.
3. The README's dataset introduction describes only the original Kaggle source; the current eight-class model uses an expanded two-source collection. Historical audit text also mentions a Grad-CAM output image that is no longer present.
4. `tools/finalise_review_revision.py` currently copies deliverables to `~/Downloads`. It was not executed today. Change that output destination before rerunning it under the current ML-folder-only instruction.
5. The paper finalisation script imports PyMuPDF, which is installed locally but not declared in `requirements.txt`; a fresh environment cannot reproduce that step from the requirements file alone.
6. Calibration, external multi-printer validation, recording-grouped splitting, a same-split backbone comparison, and a temporal-window ablation remain research gaps. The revised paper explicitly discloses these limitations.
7. Physical camera/printer operation and long-duration stability still need deployment checks. Today's HTTP/inference smoke test does not establish those properties.

## Launch

```bash
cd /home/ron/ml/fdm_defect_detection
.venv/bin/python src/live_dashboard.py
```

Open `http://127.0.0.1:8765`. Stop the server with Ctrl+C after the demonstration.

## Overall Condition

The retained eight-class inference model and local upload HTTP path are operational. The project is suitable for continuing the demonstration workflow, subject to a physical camera check. The reviewer revision is present and structurally valid, but remains uncommitted. The project should be described as a working operator-assistance prototype with disclosed research validation gaps, rather than an independently validated production printer controller.
