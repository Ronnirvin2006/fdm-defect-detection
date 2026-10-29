"use strict";
const $ = id => document.getElementById(id);
const displayName = name => (name || "").replaceAll("_", " ");
let mode = "live", stream = null, cameraGeneration = 0, loopGeneration = 0;
let activeRequest = null, inFlight = false, timer = null, photo = null, photoSource = "Photo";
let videoUrl = null, photoUrl = null, lastBlob = null, lastResult = null;
let records = [], events = [], rolling = [], lastEventKey = "", totalFrames = 0;
let previousCompletion = 0, completedTimes = [], explainNext = false;
const canvas = $("capture-canvas");
const camera = $("camera-preview"), clip = $("clip-preview");

function icons() { lucide.createIcons(); }
function toast(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  setTimeout(() => { $("toast").hidden = true; }, 3500);
}
function message(text, error = false) {
  $("source-message").textContent = text;
  $("source-message").classList.toggle("error", error);
}
function sourceStatus(text, running = false) {
  $("source-status").replaceChildren();
  const dot = document.createElement("span"); dot.className = "dot";
  $("source-status").append(dot, document.createTextNode(text));
  $("source-status").classList.toggle("running", running);
}
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("fdm-theme", theme); } catch (_) { /* Storage may be disabled. */ }
  $("theme").innerHTML = `<i data-lucide="${theme === "dark" ? "sun" : "moon"}"></i>`;
  $("theme").title = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  icons();
}
let savedTheme;
try { savedTheme = localStorage.getItem("fdm-theme"); } catch (_) { /* Use light by default. */ }
applyTheme(savedTheme === "dark" ? "dark" : "light");
$("theme").onclick = () => applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");

function resetDecision() {
  rolling = []; lastResult = null; lastBlob = null; explainNext = false;
  $("decision").textContent = "Waiting for a print";
  $("decision-detail").textContent = "No analysis yet";
  $("decision-badge").textContent = "Idle"; $("decision-badge").className = "badge neutral";
  $("confidence").textContent = "—"; $("confidence-fill").style.width = "0%";
  $("explain").disabled = true; $("explanation").hidden = true;
  setList("causes", [], "Available after a confirmed result");
  setList("actions", [], "No action recommended yet");
  $("class-scores").replaceChildren();
  completedTimes = []; previousCompletion = 0;
}
function cancelAnalysis() {
  loopGeneration++;
  clearTimeout(timer);
  if (activeRequest) activeRequest.abort();
  activeRequest = null; inFlight = false;
  rolling = []; completedTimes = [];
}
function stopCamera(reason = "Camera stopped") {
  cameraGeneration++;
  cancelAnalysis();
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null; camera.srcObject = null;
  $("start-camera").disabled = false; $("start-camera").hidden = false;
  $("stop-camera").hidden = true;
  $("snapshot").disabled = true; $("confirm-live").disabled = true; $("confirm-live").checked = false;
  $("camera-empty").hidden = false; $("camera-tag").hidden = true; $("frame-guide").hidden = true;
  if (mode === "live") { sourceStatus("Offline"); message(reason); }
}
function changeMode(next) {
  if (next === mode) return;
  stopCamera(); clip.pause(); cancelAnalysis(); resetDecision();
  mode = next;
  document.querySelectorAll(".tab").forEach(tab => {
    const selected = tab.dataset.mode === mode;
    tab.classList.toggle("active", selected); tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  ["live", "image", "video"].forEach(name => { $("panel-" + name).hidden = name !== mode; });
  $("view-title").textContent = {live:"Live camera", image:"Photo inspection", video:"Video review"}[mode];
  sourceStatus("Ready"); message("Ready");
}
document.querySelectorAll(".tab").forEach(tab => {
  tab.onclick = () => changeMode(tab.dataset.mode);
  tab.onkeydown = event => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const tabs = [...document.querySelectorAll(".tab")];
    const index = (tabs.indexOf(tab) + (event.key === "ArrowRight" ? 1 : 2)) % 3;
    changeMode(tabs[index].dataset.mode); tabs[index].focus();
  };
});
async function startCamera() {
  stopCamera(); resetDecision();
  const generation = cameraGeneration;
  $("start-camera").disabled = true;
  message("Waiting for camera permission…");
  try {
    if (!isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error("Camera access needs the HTTPS link in Chrome or Safari.");
    const selected = $("camera-select").value;
    const choice = ["environment", "user"].includes(selected) ? {facingMode:{ideal:selected}} : {deviceId:{exact:selected}};
    const opened = await navigator.mediaDevices.getUserMedia({video:{...choice, width:{ideal:640}, height:{ideal:480}, frameRate:{ideal:30, max:30}}, audio:false});
    if (generation !== cameraGeneration || document.hidden) { opened.getTracks().forEach(track => track.stop()); return; }
    stream = opened; camera.srcObject = opened; await camera.play();
    if (generation !== cameraGeneration) return;
    $("camera-empty").hidden = true; $("camera-tag").hidden = false;
    $("start-camera").hidden = true; $("stop-camera").hidden = false;
    $("snapshot").disabled = false; $("confirm-live").disabled = false; $("frame-guide").hidden = false;
    sourceStatus("Preview", true); message("Confirm the printer view to begin inspection.");
    try {
      const current = opened.getVideoTracks()[0].getSettings().deviceId;
      const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === "videoinput" && device.deviceId);
      if (devices.length) {
        $("camera-select").replaceChildren();
        devices.forEach((device, index) => {
          const option = new Option(device.label || "Camera " + (index + 1), device.deviceId);
          option.selected = device.deviceId === current; $("camera-select").add(option);
        });
      }
    } catch (_) { /* Preview remains usable when device enumeration is restricted. */ }
  } catch (error) {
    if (generation !== cameraGeneration) return;
    stopCamera();
    const messages = {NotAllowedError:"Camera permission denied. Allow camera access in browser settings.", NotFoundError:"No camera found.", NotReadableError:"Camera is busy. Close other camera apps and try again.", OverconstrainedError:"Selected camera is unavailable. Reload and select another camera."};
    message(messages[error.name] || error.message, true);
  }
}
$("start-camera").onclick = startCamera;
$("stop-camera").onclick = () => stopCamera();
$("camera-select").onchange = () => { if (stream) startCamera(); };
$("confirm-live").onchange = () => {
  cancelAnalysis(); resetDecision();
  if ($("confirm-live").checked && stream) { sourceStatus("Inspecting", true); schedule(0); }
  else { sourceStatus("Preview", !!stream); message("Inspection paused until the printer view is confirmed."); }
};
function continuousActive() {
  return !document.hidden && ((mode === "live" && stream && $("confirm-live").checked) || (mode === "video" && !clip.paused && !clip.ended && $("confirm-video").checked));
}
function schedule(delay = 35) {
  clearTimeout(timer);
  if (continuousActive() && !inFlight) timer = setTimeout(() => analyseCurrent(), delay);
}
async function frameBlob(element, limit = 480) {
  const width = element.videoWidth || element.naturalWidth, height = element.videoHeight || element.naturalHeight;
  if (!width || !height) throw new Error("Waiting for a readable frame.");
  const scale = Math.min(1, limit / Math.max(width, height));
  canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
  canvas.getContext("2d", {alpha:false}).drawImage(element, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("Unable to capture frame")), "image/jpeg", .72));
}
async function analyseCurrent(single = false) {
  if (inFlight || (!single && !continuousActive())) return;
  inFlight = true;
  const generation = loopGeneration;
  const controller = new AbortController(); activeRequest = controller;
  const timeout = setTimeout(() => controller.abort(), 30000);
  let retry = false;
  const source = mode === "live" ? "Live camera" : mode === "video" ? "Video" : photoSource;
  const mediaTime = mode === "video" ? clip.currentTime : null;
  const explain = explainNext; explainNext = false;
  if (single) { message(explain ? "Generating explanation…" : "Analysing photo…"); $("analyse-photo").disabled = true; }
  const started = performance.now();
  try {
    const element = mode === "live" ? camera : mode === "video" ? clip : photo;
    const blob = await frameBlob(element, mode === "image" ? 1280 : 480);
    if (generation !== loopGeneration) return;
    const response = await fetch(explain ? "/api/explain" : mode === "image" ? "/api/image" : "/api/frame", {
      method:"POST", body:blob, headers:{"Content-Type":"image/jpeg"}, signal:controller.signal, cache:"no-store"
    });
    const result = await response.json();
    if (generation !== loopGeneration) return;
    if (result.busy) { retry = true; message("Analysis busy — reconnecting to the latest frame."); return; }
    if (!response.ok || result.error) throw new Error(result.feedback || "Frame analysis failed.");
    const now = performance.now(), roundtrip = now - started;
    totalFrames++;
    completedTimes.push(now); completedTimes = completedTimes.filter(t => t >= now - 5000);
    const rate = completedTimes.length >= 2 ? (completedTimes.length - 1) * 1000 / (now - completedTimes[0]) : 0;
    metric("metric-rate", rate ? rate.toFixed(1) : "—", " checks/s");
    metric("metric-processing", Math.round(result.processing_ms), " ms");
    metric("metric-latency", Math.round(roundtrip), " ms");
    metric("metric-frames", totalFrames, "");
    previousCompletion = now;
    lastBlob = blob; lastResult = result;
    const decision = classify(result, !single);
    renderResult(result, decision);
    record(result, decision, source, mediaTime, roundtrip);
    if (result.explanation) {
      $("heatmap").src = result.explanation; $("download-heatmap").href = result.explanation; $("explanation").hidden = false;
    }
    sourceStatus(single ? "Analysed" : "Inspecting", !single);
    message("Updated " + new Date().toLocaleTimeString());
    $("connection").classList.add("online"); $("connection").lastElementChild.textContent = "Connected";
  } catch (error) {
    if (generation !== loopGeneration) return;
    retry = true;
    message(error.name === "AbortError" ? "Analysis timed out. Retrying with the latest frame…" : error.message, true);
    $("connection").classList.remove("online"); $("connection").lastElementChild.textContent = "Reconnecting";
  } finally {
    clearTimeout(timeout);
    if (generation === loopGeneration) {
      inFlight = false; activeRequest = null;
      $("analyse-photo").disabled = !photo || !$("confirm-image").checked;
      if (mode === "image" && explainNext) setTimeout(() => analyseCurrent(true), 35);
      else schedule(retry ? 600 : 35);
    }
  }
}
function classify(result, temporal) {
  if (result.status === "rejected") { rolling = []; return {status:"rejected", label:"Scene rejected", detail:result.feedback}; }
  if (!temporal) return {status:result.status, label:result.status === "uncertain" ? "Uncertain" : displayName(result.class_name), detail:result.status === "uncertain" ? "No reliable defect decision" : result.description};
  rolling.push({name:result.class_name, confidence:result.confidence}); rolling = rolling.slice(-3);
  const stable = rolling.length === 3 && rolling.every(item => item.name === result.class_name && item.confidence >= 70);
  if (!stable) return {status:result.confidence < 70 ? "uncertain" : "checking", label:result.confidence < 70 ? "Uncertain" : "Checking print", detail:"Awaiting a consistent reading"};
  return {status:result.class_name === "No_defect" ? "normal" : "defect", label:displayName(result.class_name), detail:result.description};
}
function metric(id, value, unit) { $(id).replaceChildren(document.createTextNode(value)); const small = document.createElement("small"); small.textContent = unit; $(id).append(small); }
function setList(id, items, placeholder) {
  $(id).replaceChildren();
  (items.length ? items.slice(0, 3) : [placeholder]).forEach(text => { const li = document.createElement("li"); li.textContent = text; if (!items.length) li.className = "placeholder"; $(id).append(li); });
}
function renderResult(result, decision) {
  $("decision").textContent = decision.label;
  $("decision-detail").textContent = decision.detail;
  $("decision-badge").className = "badge " + decision.status;
  $("decision-badge").textContent = {normal:"Normal", defect:"Defect alert", uncertain:"Uncertain", rejected:"View rejected", checking:"Confirming"}[decision.status];
  $("confidence").textContent = result.status === "rejected" ? "—" : result.confidence.toFixed(1) + "%";
  $("confidence-fill").style.width = (result.confidence || 0) + "%";
  $("confidence-fill").style.background = decision.status === "defect" ? "var(--red)" : "var(--accent)";
  const actionable = ["defect", "normal"].includes(decision.status);
  setList("causes", actionable ? result.causes || [] : [], decision.status === "rejected" ? result.feedback : "No confirmed defect cause");
  setList("actions", actionable ? result.actions || [] : [], decision.status === "rejected" ? "Point the camera at the print, without a person in view" : "Keep the print visible and wait for a stable reading");
  $("class-scores").replaceChildren();
  [...result.probabilities || []].sort((a,b) => b.value - a.value).forEach(item => {
    const row = document.createElement("div"); row.className = "score-row";
    const name = document.createElement("span"); name.textContent = displayName(item.name);
    const score = document.createElement("span"); score.textContent = item.value.toFixed(1) + "%";
    const bar = document.createElement("progress"); bar.max = 100; bar.value = item.value;
    row.append(name, score, bar); $("class-scores").append(row);
  });
  $("result-time").textContent = new Date().toLocaleTimeString();
  $("explain").disabled = result.status === "rejected";
}
function record(result, decision, source, mediaTime, roundtrip) {
  const row = {timestamp:new Date().toISOString(), source, video_time_s:mediaTime == null ? "" : mediaTime.toFixed(3), predicted_class:result.class_name || "", confidence_pct:result.confidence || 0, displayed_condition:decision.label, status:decision.status, scene_reason:result.scene?.reason || "", processing_ms:result.processing_ms, roundtrip_ms:Math.round(roundtrip), feedback:result.feedback || ""};
  records.push(row); records = records.slice(-10000);
  const key = source + ":" + decision.status + ":" + decision.label;
  if (key !== lastEventKey) { lastEventKey = key; events.unshift(row); events = events.slice(0, 10); renderHistory(); }
  $("history-count").textContent = records.length + " checks";
  $("export").disabled = false; $("clear-history").disabled = false;
}
function renderHistory() {
  $("history-body").replaceChildren();
  events.forEach(row => {
    const tr = document.createElement("tr"); tr.className = "history-row";
    [new Date(row.timestamp).toLocaleTimeString(), row.source, row.displayed_condition, row.status === "rejected" ? "—" : Number(row.confidence_pct).toFixed(1) + "%", row.status].forEach(text => { const td = document.createElement("td"); td.textContent = text; tr.append(td); });
    $("history-body").append(tr);
  });
}
function download(blob, name) {
  const url = URL.createObjectURL(blob), a = document.createElement("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("export").onclick = () => {
  if (!records.length) return;
  const columns = Object.keys(records[0]);
  const cell = value => { const text = String(value ?? ""); return '"' + (/^[=+@-]/.test(text) ? "'" + text : text).replaceAll('"','""') + '"'; };
  const lines = [columns.map(cell).join(","), ...records.map(row => columns.map(column => cell(row[column])).join(","))];
  download(new Blob(["\ufeff" + lines.join("\r\n")], {type:"text/csv;charset=utf-8"}), "fdm_inspection_" + new Date().toISOString().slice(0,19).replaceAll(":","-") + ".csv");
};
$("clear-history").onclick = () => {
  if (!confirm("Clear this session's inspection activity?")) return;
  records = []; events = []; lastEventKey = "";
  $("history-body").innerHTML = '<tr><td colspan="5" class="empty-history">No inspection activity</td></tr>';
  $("history-count").textContent = "0"; $("export").disabled = true; $("clear-history").disabled = true;
};
async function loadPhoto(blob, name, source = "Photo") {
  cancelAnalysis(); resetDecision();
  if (blob.size > 25 * 1024 * 1024) { message("Photo must be smaller than 25 MB.", true); return; }
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(blob);
  const image = new Image();
  try {
    image.src = photoUrl; await image.decode(); photo = image; photoSource = source;
    $("photo-preview").src = photoUrl; $("photo-surface").hidden = false; $("image-drop").hidden = true;
    $("replace-photo").hidden = false; $("image-name").textContent = name;
    $("confirm-image").disabled = false; $("confirm-image").checked = false; $("analyse-photo").disabled = true;
    message("Confirm the print view, then analyse the photo.");
  } catch (_) { message("Photo could not be opened. Choose a JPG, PNG or WebP.", true); }
}
$("image-file").onchange = event => { if (event.target.files[0]) loadPhoto(event.target.files[0], event.target.files[0].name); };
$("replace-photo").onclick = () => $("image-file").click();
$("confirm-image").onchange = () => { $("analyse-photo").disabled = !photo || !$("confirm-image").checked; };
$("analyse-photo").onclick = () => analyseCurrent(true);
$("snapshot").onclick = async () => {
  try {
    const blob = await frameBlob(camera, 1280);
    changeMode("image"); await loadPhoto(blob, "Camera snapshot", "Snapshot"); toast("Snapshot captured");
  } catch (error) { message(error.message, true); }
};
function loadVideo(file) {
  cancelAnalysis(); resetDecision(); clip.pause();
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(file); clip.src = videoUrl;
  $("clip-surface").hidden = false; $("video-drop").hidden = true; $("replace-video").hidden = false;
  $("video-name").textContent = file.name; $("confirm-video").checked = false;
  $("confirm-video").disabled = true; $("analyse-video").disabled = true;
  message("Opening video…");
}
$("video-file").onchange = event => { if (event.target.files[0]) loadVideo(event.target.files[0]); };
$("replace-video").onclick = () => $("video-file").click();
clip.onloadeddata = () => { $("confirm-video").disabled = false; message("Confirm the printer view to analyse the video."); };
clip.onerror = () => message("Video codec is not supported by this browser. Try an H.264 MP4.", true);
$("confirm-video").onchange = () => {
  $("analyse-video").disabled = !$("confirm-video").checked; cancelAnalysis(); rolling = [];
  if ($("confirm-video").checked && !clip.paused) schedule(0);
};
$("analyse-video").onclick = () => { clip.play().catch(error => message(error.message, true)); };
clip.onplay = () => { if ($("confirm-video").checked) { sourceStatus("Analysing video", true); schedule(0); } };
clip.onpause = () => { cancelAnalysis(); sourceStatus(clip.ended ? "Video complete" : "Paused"); };
clip.onseeking = () => { cancelAnalysis(); rolling = []; };
clip.onseeked = () => schedule(0);
clip.onended = () => { cancelAnalysis(); sourceStatus("Video complete"); message("Video complete. Inspection activity is ready to export."); };
$("explain").onclick = () => {
  explainNext = true;
  if (mode === "image") analyseCurrent(true);
  else if (continuousActive()) { message("Explaining the next frame…"); schedule(0); }
  else { explainNext = false; message("Resume the camera or video to explain a current frame."); }
};
function bindDropzone(id, loader) {
  const zone = $(id);
  zone.ondragover = event => { event.preventDefault(); zone.classList.add("dragging"); };
  zone.ondragleave = () => zone.classList.remove("dragging");
  zone.ondrop = event => { event.preventDefault(); zone.classList.remove("dragging"); if (event.dataTransfer.files[0]) loader(event.dataTransfer.files[0]); };
}
bindDropzone("image-drop", file => loadPhoto(file, file.name)); bindDropzone("video-drop", loadVideo);
document.addEventListener("visibilitychange", () => { if (document.hidden) { stopCamera("Camera paused while page is hidden"); clip.pause(); } });
window.addEventListener("pagehide", () => { stopCamera(); clip.pause(); });
async function connect() {
  try {
    const response = await fetch("/api/info", {cache:"no-store"});
    if (!response.ok) throw new Error("Model unavailable");
    const info = await response.json();
    $("connection").classList.add("online"); $("connection").lastElementChild.textContent = "Model ready";
    $("model-id").textContent = info.model_id.slice(0, 6);
  } catch (_) { $("connection").lastElementChild.textContent = "Offline"; message("Unable to connect to the model. Check the host connection.", true); setTimeout(connect, 5000); }
}
icons(); connect();
