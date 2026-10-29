# Publish FDM Defect Detection Online

The final polished workspace is now the direct Python dashboard described in
`README.md` and `REMOTE_PHONE_TESTING.md`. Use `bash tools/serve_remote.sh` for the
remote-phone test. This document is retained for the older optional Streamlit
interface; deploying `src/app.py` does not serve the new direct dashboard.

The application is a Python/TensorFlow Streamlit app. Use Streamlit Community Cloud for the public hosted version. A static website host cannot load the Keras model or run the image inference code.

## One-time deployment

1. Open `https://share.streamlit.io` and sign in with the GitHub account that owns `Ronnirvin2006/fdm-defect-detection`.
2. Choose **Create app**.
3. Select repository `Ronnirvin2006/fdm-defect-detection`.
4. Select branch `main`.
5. Set the main file to `src/app.py`.
6. Choose an app subdomain such as `fdm-defect-detection`.
7. Open **Advanced settings** and select Python **3.13**. TensorFlow 2.20.0 does not provide a compatible wheel for Python 3.14.
8. Deploy.

If the app was already created with Python 3.14, Streamlit requires deleting and redeploying the app to change its Python version. Select Python 3.13 during the new deployment. The committed `runtime.txt` records the intended version, but still verify the dropdown because Streamlit's deployment settings take precedence.

The public address will be similar to `https://fdm-defect-detection.streamlit.app`. The exact address is assigned by Streamlit.

The app has a deployment-specific `src/requirements.txt`. Because the entrypoint is `src/app.py`, Streamlit can use this smaller dependency file instead of installing the training and Kaggle tooling from the repository root.

## What visitors can use

- Upload a JPG, JPEG, PNG, BMP, or WEBP image.
- Upload a short MP4, MOV, AVI, or WEBM video and analyse evenly spaced frames across the clip.
- Receive one of eight predictions: Cracking, Layer shifting, No defect, Off platform, Spaghetti, Stringing, Under-extrusion, or Warping.
- View the class confidence chart and curated causes/corrective checks.
- Use the **Live camera** tab for a live browser preview with automatic sampled-frame inference. The browser must grant camera permission.
- Use the camera snapshot tab.
- Keep a session inspection history and download it as CSV from the sidebar.

The hosted Streamlit app accesses the visitor's camera in the browser and sends compressed JPEG frames through Streamlit's existing connection. There is no WebRTC peer connection and no TURN requirement. The continuous local dashboard remains available with `src/live_dashboard.py` for a computer physically connected to a printer camera. See `REMOTE_PHONE_TESTING.md` to run inference on your laptop and share an HTTPS testing link across different networks.

Video analysis samples at most 36 frames and averages their class probabilities. It reports the overall condition, frame-by-frame sampled predictions, observed classes, confidence, and the same curated feedback used for images. It is a clip-level classification feature; it does not localise a defect, guarantee that a short transient was captured, or replace a validated real-time alarm.

## Important deployment notes

- No Kaggle credentials or OctoPrint API keys are needed for the public image-classification app.
- Live camera access requires HTTPS (or localhost), a supported browser, and camera permission. The app prefers the rear camera, offers camera selection, and stops capture when hidden or closed.
- Leave Secrets empty for this camera transport. Old Cloudflare TURN secrets are no longer used.
- Do not add `.kaggle/access_token` or printer API keys to GitHub.
- Uploaded images are processed by the hosted app runtime. Do not upload confidential images.
- The app provides decision support. It does not move printer axes, change heaters, or stop a printer automatically.
- Search-engine discovery is separate from deployment. Search engines may take time to index a new app; typing “FDM detection” will not guarantee it appears first.

## Local check

```bash
cd /home/ron/ml/fdm_defect_detection
.venv/bin/streamlit run src/app.py
```

Then open `http://localhost:8501`.
