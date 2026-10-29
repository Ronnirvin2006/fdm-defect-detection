# Remote Phone Test

## Host

```bash
cd /home/ron/ml/fdm_defect_detection
bash tools/serve_remote.sh
```

Share the printed HTTPS URL. Keep the laptop awake, online and plugged in. Close
training first so GPU inference stays responsive. The first run downloads the
tunnel executable into `.tools/` if needed. No router forwarding, shared Wi-Fi,
Streamlit or TURN account is required. Ctrl+C stops the server and tunnel.
Logs are in `logs/remote_app.log`. If the port is busy, run
`FDM_REMOTE_PORT=8503 bash tools/serve_remote.sh`.

## Friend

1. Open the HTTPS link directly in Android Chrome or iPhone Safari, not an embedded messaging browser.
2. Press **Start camera**, then allow permission. Previously granted permission may be reused without another prompt.
3. Rear camera is preferred; select another camera/lens if needed.
4. Frame the actual part/build plate, then tick **Printer and print area are in view**. Inference does not run before confirmation.
5. The preview is continuous. Each response schedules the newest frame. Three agreeing scores of at least 70% confirm the displayed condition.
6. Review causes/checks beside the view. Preliminary face/lighting checks cannot guarantee rejection of every unrelated object.
7. Use **Photo** for upload/drop, **Video** for playback analysis, the camera icon for a snapshot, and **Explain frame** for attention.
8. **Export CSV** before closing to retain up to 10,000 recent checks.
9. Stop, source switching, hiding or closing releases the camera. Press Start to resume after returning.

## Privacy and Limits

- The temporary link is public to anyone who has it, changes on restart, and has no uptime guarantee. Share privately for testing.
- Compressed frames pass through Cloudflare to the host. This workflow does not write live frames or uploads to disk.
- This is a whole-image classifier, not a nozzle/bed detector or multi-defect localiser. Confidence is not calibrated.
- No temperature, axis, cleaning or printer-pause command is enabled.
- Real Android/iOS hardware and printer accuracy need your friend's test. Automated QA uses simulated media and responsive Chromium viewports.
- Poor uplink/host load lowers analysis rate; local preview remains independent. Actual checks/s and latency are shown.
- Permanent availability requires an always-on host or a managed tunnel/domain.

## Field Evidence

Save the CSV and record printer, filament, lighting, phone/browser and viewpoint.
Capture normal printing and any safely available known defect. Never damage or
overheat a printer to create test data. Record false alarms and missed visible
defects, not only confidence scores. These observations must remain separate
from historical training results in the paper.
