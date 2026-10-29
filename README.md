# FDM Inspect

An eight-condition Fused Deposition Modelling inspection workspace. A remote
phone/laptop supplies images; the host runs EfficientNetB0 and returns class
scores, curated causes and recommended checks. This is whole-image classification
and operator decision support, not defect localisation or autonomous control.

## Launch and Share

```bash
cd /home/ron/ml/fdm_defect_detection
bash tools/serve_remote.sh
```

Share the printed **HTTPS `trycloudflare.com` link**. The phone and host can use
different networks. Keep the host awake, online and plugged in. Anyone with the
link can access it; this temporary testing URL changes when restarted.

For local use without a public tunnel:

```bash
.venv/bin/python src/remote_camera_server.py --port 8502
```

Open `http://localhost:8502`. A phone cannot use the laptop's localhost address;
remote camera permission needs the HTTPS link. More details: `REMOTE_PHONE_TESTING.md`.

## Inspection

- **Live camera:** Start, grant permission, choose a rear camera and confirm the
  print area. Preview is independent of inference. Each response requests a fresh
  frame, so there is no backlog of old video frames.
- **Photo:** JPG, PNG or WebP upload/drop with view confirmation.
- **Video:** continuous analysis during browser playback with video timestamps.
  Codec support depends on the browser; H.264 MP4 is a useful interoperable
  option. Not every encoded video frame is analysed.
- **Snapshot:** captures the current camera view and opens photo inspection.
- **Feedback:** confidence, possible causes, recommended checks and class scores.
  Three agreeing predictions at 70% confidence confirm a live condition. These
  thresholds are uncalibrated heuristics, not safety guarantees.
- **Explain frame:** on-demand Grad-CAM attention, not a segmented defect.
- **CSV:** exports up to 10,000 recent checks with timestamps, source, scores,
  displayed state and latency. Export before closing the page.
- **Light/dark theme**, camera selection and automatic release on Stop, source
  change, page hide or page exit.

Preliminary checks reject detected faces, extreme lighting and nearly uniform
views. They are **not trained printer/nozzle detection**; unrelated objects can
still pass. Confirm the actual print and visually review important alerts. No
temperature, speed or axis value is invented. No machine command is enabled.

## Active Model and Evidence

`models/best_model.keras` is the single active checkpoint, selected from
`weak_classes_20261029_183217.keras`. Provenance and checksum are in
`models/deployment.json`. All earlier weights remain locally; the original model
is backed up as `models/historical_20260727.keras`. Other checkpoints and source
images are intentionally excluded from Git.

Classes: Cracking, Layer shifting, No defect, Off platform, Spaghetti, Stringing,
Under-extrusion and Warping.

| Experiment | Evaluation | Accuracy | Macro F1 |
|---|---|---:|---:|
| Historical model | 661-image stratified test | 96.52% | 96.95% |
| Selected continuation | 4,427-image reused grouped validation | 83.83% | 83.57% |

These are **not directly comparable**. Grouped validation was reused for model
selection and may overlap source-checkpoint training history. It is not an
independent test; external printer validation is pending.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/check_final_workspace.py --url http://localhost:8502
```

The browser QA helper needs Playwright and an installed Chrome; runtime camera
use does not. Short simulated-camera checks measured about 11.4-11.9 checks/s
locally and 2.9-3.6 checks/s through public HTTPS in desktop/mobile-sized Chromium
viewports. These are software checks, not actual Android/iOS or physical-printer
validation. Evidence is retained in `outputs/reports/release_validation.json`.

For a new environment, install the pinned `requirements.txt` in Python 3.10-3.13.
CUDA/GPU configuration is a separate host setup step.

## IAME Paper

Current files are only in `outputs/paper/IAME2026_Final/`:

- `IAME2026_Paper54_Camera_Ready.pdf`: six-page A4 manuscript.
- `IAME2026_Paper54_Full_Paper.pdf`: identical manuscript, alternate filename.
- `IAME2026_Paper54_Response_to_Reviewers.pdf`: two-page response.
- Editable DOCX/Markdown, project-derived vector figures, evidence and ZIP.

The paper separates historical and current results. Calibration diagnostics use
reused validation probabilities, not independent test data. An under-10%
similarity score cannot be certified without Turnitin/iThenticate or another
institution-approved report. The acceptance email specifies less than 15% and
camera-ready completion by 31 October 2026. Supervisor approval, registration and
CMT upload remain author actions.

## Files

| Path | Purpose |
|---|---|
| `src/remote_camera_server.py` | Main direct-inference web server |
| `src/web/` | Dashboard, themes, camera/upload logic and local Lucide icons |
| `src/scene_guard.py` | Preliminary unsuitable-view checks |
| `src/gradcam.py` | Attention overlays and CLI |
| `src/defect_knowledge.py` | Fixed troubleshooting knowledge |
| `src/finetune_local.py` | Offline continuation training |
| `src/app.py` | Optional legacy Streamlit interface, not the final dashboard |
| `src/live_dashboard.py` | Earlier local OpenCV/telemetry prototype |
| `tools/serve_remote.sh` | Model server plus public testing tunnel |
| `tools/build_final_submission.py` | Rebuild final editable paper package |
| `models/` | Active model, class mapping and preserved local checkpoints |
| `data/` | Retained datasets and prepared images, excluded from Git |
| `outputs/reports/` | Training and evaluation evidence |
| `outputs/paper/IAME2026_Final/` | Current submission package |
| `tests/` | Training, input-validation and server tests |

`DEPLOY_STREAMLIT.md` documents the optional fallback. The direct Python dashboard
is recommended for the remote-camera test. A permanent URL requires an always-on
HTTPS host or a managed tunnel/domain; the laptop testing link is temporary.
