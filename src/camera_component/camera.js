"use strict";
const video = document.getElementById("video");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const selector = document.getElementById("camera");
const statusNode = document.getElementById("status");
const label = document.getElementById("label");
const feedback = document.getElementById("feedback");
const canvas = document.getElementById("canvas");
const direct = !!document.querySelector('meta[name="fdm-transport"][content="http"]');
let activeRequest = null;
let stream = null, timer = null, watchdog = null, pending = null, generation = 0;
let interval = 40;
let sentAt = 0, completed = 0, lastCompletedAt = 0;
const session = Date.now().toString(36) + Math.random().toString(36).slice(2);
let sequence = 0;
function send(type, data = {}) {
  if (direct) return;
  window.parent.postMessage({isStreamlitMessage: true, type, ...data}, "*");
}
function resize() { send("streamlit:setFrameHeight", {height: document.body.scrollHeight + 8}); }
function status(message, error = false) {
  statusNode.textContent = message;
  statusNode.className = error ? "error" : "";
  resize();
}
function stopCamera(message = "Camera stopped") {
  generation++;
  clearTimeout(timer); clearTimeout(watchdog);
  if (activeRequest) activeRequest.abort();
  activeRequest = null;
  pending = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null; video.srcObject = null;
  startButton.disabled = false; stopButton.disabled = true;
  label.textContent = message; feedback.textContent = "";
  status(message);
}
function schedule() {
  clearTimeout(timer);
  if (stream && !pending && !document.hidden) timer = setTimeout(capture, interval);
}
function capture() {
  if (!stream || pending || document.hidden) return;
  if (!video.videoWidth || video.readyState < 2) { schedule(); return; }
  const scale = Math.min(1, 480 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  pending = session + ":" + (++sequence);
  sentAt = performance.now();
  if (!completed) status("Connecting to defect analysis...");
  // Keep one request in flight, so slow inference cannot create a frame backlog.
  const id = pending;
  if (direct) {
    const controller = new AbortController();
    activeRequest = controller;
    canvas.toBlob(async blob => {
      if (pending !== id || !stream) return;
      try {
        const response = await fetch("/api/frame", {
          method:"POST", headers:{"Content-Type":"image/jpeg"}, body:blob,
          signal:controller.signal, cache:"no-store"
        });
        const result = await response.json();
        receiveResult({...result, id});
      } catch (error) {
        if (error.name !== "AbortError") receiveResult({id, error:true, label:"Connection interrupted", feedback:"Reconnecting to analysis..."});
      }
    }, "image/jpeg", .65);
  } else {
    send("streamlit:setComponentValue", {dataType:"json", value:{id, image:canvas.toDataURL("image/jpeg", .65)}});
  }
  watchdog = setTimeout(() => {
    stopCamera("Analysis timed out. Restart the camera or reload the page.");
  }, 60000);
}
async function startCamera() {
  stopCamera();
  const request = generation;
  startButton.disabled = true;
  status("Waiting for camera permission...");
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Open the HTTPS link in Chrome or Safari to allow camera access.");
    }
    const selected = selector.value;
    const constraints = selected === "environment" || selected === "user"
      ? {facingMode:{ideal:selected}, width:{ideal:640}, height:{ideal:480}, frameRate:{ideal:30,max:30}}
      : {deviceId:{exact:selected}, width:{ideal:640}, height:{ideal:480}, frameRate:{ideal:30,max:30}};
    const opened = await navigator.mediaDevices.getUserMedia({video:constraints, audio:false});
    if (request !== generation || document.hidden) { opened.getTracks().forEach(t => t.stop()); return; }
    stream = opened; video.srcObject = opened;
    await video.play();
    stopButton.disabled = false;
    label.textContent = "Waiting for first analysis";
    status("Camera connected");
    const devices = await navigator.mediaDevices.enumerateDevices();
    selector.replaceChildren();
    const current = opened.getVideoTracks()[0].getSettings().deviceId;
    devices.filter(d => d.kind === "videoinput").forEach((device, index) => {
      const option = new Option(device.label || "Camera " + (index + 1), device.deviceId);
      option.selected = device.deviceId === current;
      selector.add(option);
    });
    schedule();
  } catch (error) {
    if (request !== generation) return;
    stopCamera();
    const messages = {
      NotAllowedError:"Camera permission denied. Allow camera access in this site's browser settings.",
      NotFoundError:"No camera found on this device.",
      NotReadableError:"Camera is busy. Close other camera apps and try again.",
      OverconstrainedError:"Selected camera is unavailable. Reload and try another camera."
    };
    status(messages[error.name] || error.message, true);
  }
}
startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", () => stopCamera());
selector.addEventListener("change", () => { if (stream) startCamera(); });
window.addEventListener("pagehide", () => stopCamera());
document.addEventListener("visibilitychange", () => { if (document.hidden) stopCamera("Camera paused while page is hidden"); });
function receiveResult(result) {
  if (result && result.id === pending) {
    clearTimeout(watchdog); pending = null;
    activeRequest = null;
    const now = performance.now();
    const roundtrip = Math.round(now - sentAt);
    const rate = lastCompletedAt ? (1000 / (now - lastCompletedAt)).toFixed(1) : "--";
    lastCompletedAt = now;
    completed++;
    label.textContent = result.label;
    feedback.textContent = result.feedback || "";
    status(result.error ? result.feedback : "Updated " + new Date().toLocaleTimeString() + " | " + rate + " checks/s | " + roundtrip + " ms round trip", !!result.error);
    schedule();
  }
  resize();
}
window.addEventListener("message", event => {
  if (event.source !== window.parent || event.data?.type !== "streamlit:render") return;
  const args = event.data.args || {};
  interval = Math.max(40, args.interval_ms || 40);
  receiveResult(args.result);
});
video.addEventListener("loadedmetadata", resize);
window.addEventListener("resize", resize);
send("streamlit:componentReady", {apiVersion:1});
resize();
