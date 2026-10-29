# Publish FDM Defect Detection Online

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
- Use the **Live Webcam** tab for continuous browser-camera inference. The browser must grant camera permission.
- Use the camera snapshot tab.

The hosted Streamlit app uses WebRTC for browser-camera access; it does not access a visitor's local OpenCV device through the server. The continuous local dashboard remains available with `src/live_dashboard.py` for a computer physically connected to a printer camera.

Video analysis samples at most 36 frames and averages their class probabilities. It reports the overall condition, frame-by-frame sampled predictions, observed classes, confidence, and the same curated feedback used for images. It is a clip-level classification feature; it does not localise a defect, guarantee that a short transient was captured, or replace a validated real-time alarm.

## Important deployment notes

- No Kaggle credentials or OctoPrint API keys are needed for the public image-classification app.
- Live WebRTC requires HTTPS, a supported browser, and camera permission. If a corporate firewall blocks WebRTC, use Upload Video or Camera Snapshot.
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
