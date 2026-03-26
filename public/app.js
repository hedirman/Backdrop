const elements = {
  videoInput: document.querySelector("#video-input"),
  engineSelect: document.querySelector("#engine-select"),
  backgroundMode: document.querySelector("#background-mode"),
  backgroundColor: document.querySelector("#background-color"),
  backgroundImage: document.querySelector("#background-image"),
  processFps: document.querySelector("#process-fps"),
  processFpsValue: document.querySelector("#process-fps-value"),
  rvmDetailControl: document.querySelector("#rvm-detail-control"),
  rvmDetail: document.querySelector("#rvm-detail"),
  rvmDetailValue: document.querySelector("#rvm-detail-value"),
  threshold: document.querySelector("#threshold"),
  thresholdValue: document.querySelector("#threshold-value"),
  feather: document.querySelector("#feather"),
  featherValue: document.querySelector("#feather-value"),
  maskStrength: document.querySelector("#mask-strength"),
  maskStrengthValue: document.querySelector("#mask-strength-value"),
  temporalSmooth: document.querySelector("#temporal-smooth"),
  temporalSmoothValue: document.querySelector("#temporal-smooth-value"),
  edgeShrink: document.querySelector("#edge-shrink"),
  edgeShrinkValue: document.querySelector("#edge-shrink-value"),
  foregroundBoost: document.querySelector("#foreground-boost"),
  foregroundBoostValue: document.querySelector("#foreground-boost-value"),
  shadow: document.querySelector("#shadow"),
  shadowValue: document.querySelector("#shadow-value"),
  startButton: document.querySelector("#start-button"),
  pauseButton: document.querySelector("#pause-button"),
  loopButton: document.querySelector("#loop-button"),
  cancelButton: document.querySelector("#cancel-button"),
  exportButton: document.querySelector("#export-button"),
  status: document.querySelector("#status"),
  finishedMeta: document.querySelector("#finished-meta"),
  engineMeta: document.querySelector("#engine-meta"),
  uploadCard: document.querySelector("#upload-card"),
  settingsPanel: document.querySelector("#settings-panel"),
  uploadTitle: document.querySelector("#upload-title"),
  uploadDetail: document.querySelector("#upload-detail"),
  tooltipLayer: document.querySelector("#tooltip-layer"),
  renderMeta: document.querySelector("#render-meta"),
  expectedDurationMeta: document.querySelector("#expected-duration-meta"),
  encodedDurationMeta: document.querySelector("#encoded-duration-meta"),
  playbackRateMeta: document.querySelector("#playback-rate-meta"),
  videoMeta: document.querySelector("#video-meta"),
  playbackMeta: document.querySelector("#playback-meta"),
  sourceVideo: document.querySelector("#source-video"),
  previewVideo: document.querySelector("#preview-video"),
  outputCanvas: document.querySelector("#output-canvas"),
  sourceCanvas: document.querySelector("#source-canvas"),
  maskCanvas: document.querySelector("#mask-canvas")
};

const outputContext = elements.outputCanvas.getContext("2d", { alpha: true });
const sourceContext = elements.sourceCanvas.getContext("2d", { willReadFrequently: true });
const maskContext = elements.maskCanvas.getContext("2d", { willReadFrequently: true });

const state = {
  videoUrl: "",
  selectedFileName: "",
  bgImage: null,
  backendReady: false,
  backendInfo: null,
  sessionId: "",
  videoReady: false,
  running: false,
  processingVideo: false,
  cancelProcessing: false,
  recording: false,
  frameHandle: 0,
  stream: null,
  recorder: null,
  recordedChunks: [],
  processingFrame: false,
  previousMaskData: null,
  subjectCanvas: null,
  subjectContext: null,
  latestMaskStats: null,
  latestFramePayload: null,
  processedFrames: [],
  processedFrameIndex: 0,
  processedFps: 24,
  processedDuration: 0,
  processedFramesDirty: false,
  previewVideoUrl: "",
  previewVideoDuration: 0,
  loopPlayback: false,
  processingStartedAt: 0
};
const seekEpsilon = 0.001;

function getSelectedEngine() {
  const value = elements.engineSelect.value;
  if (value === "rvm" || value === "coreml_quality" || value === "modnet" || value === "ppmattingv2") {
    return value;
  }
  return "coreml_preview";
}

function isCoreMlEngine(engine = getSelectedEngine()) {
  return engine === "coreml_preview" || engine === "coreml_quality";
}

function getEngineLabel(engine = getSelectedEngine()) {
  if (engine === "rvm") {
    return "RVM";
  }
  if (engine === "ppmattingv2") {
    return "PP-MattingV2";
  }
  if (engine === "modnet") {
    return "MODNet";
  }
  if (engine === "coreml_quality") {
    return "Core ML Quality";
  }
  return "Core ML Preview";
}

function syncEngineAwareControls() {
  if (elements.rvmDetailControl) {
    const visible = getSelectedEngine() === "rvm";
    elements.rvmDetailControl.hidden = !visible;
    elements.rvmDetailControl.style.display = visible ? "" : "none";
  }
}

function formatSeconds(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return "Waiting for preview";
  }
  return `${value.toFixed(2)}s`;
}

function formatElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

function updateTimingDebug() {
  const expectedDuration = state.processedFrames.length && state.processedFps
    ? state.processedFrames.length / state.processedFps
    : 0;
  const encodedDuration = state.previewVideoDuration || 0;
  const appliedPlaybackRate = elements.previewVideo.playbackRate || 1;

  if (elements.expectedDurationMeta) {
    elements.expectedDurationMeta.textContent = formatSeconds(expectedDuration);
  }
  if (elements.encodedDurationMeta) {
    elements.encodedDurationMeta.textContent = formatSeconds(encodedDuration);
  }
  if (elements.playbackRateMeta) {
    elements.playbackRateMeta.textContent = `${appliedPlaybackRate.toFixed(2)}x`;
  }
}

function bindRange(input, label, formatter = (value) => value) {
  const sync = () => {
    label.textContent = formatter(input.value);
  };

  input.addEventListener("input", sync);
  sync();
}

bindRange(elements.processFps, elements.processFpsValue);
bindRange(elements.rvmDetail, elements.rvmDetailValue, (value) => `${(Number(value) / 100).toFixed(2)}x`);
bindRange(elements.threshold, elements.thresholdValue, (value) => (Number(value) / 100).toFixed(2));
bindRange(elements.feather, elements.featherValue, (value) => `${value}px`);
bindRange(elements.maskStrength, elements.maskStrengthValue, (value) => `${(Number(value) / 100).toFixed(2)}x`);
bindRange(elements.temporalSmooth, elements.temporalSmoothValue, (value) => (Number(value) / 100).toFixed(2));
bindRange(elements.edgeShrink, elements.edgeShrinkValue, (value) => `${value}px`);
bindRange(elements.foregroundBoost, elements.foregroundBoostValue, (value) => `${(Number(value) / 100).toFixed(2)}x`);
bindRange(elements.shadow, elements.shadowValue, (value) => `${value}px`);

function setStatus(message) {
  elements.status.textContent = message;
}

function setFinishedMeta(message) {
  if (elements.finishedMeta) {
    elements.finishedMeta.textContent = message;
  }
}

function closeTooltips() {
  document.querySelectorAll(".info-icon.is-open").forEach((icon) => {
    icon.classList.remove("is-open");
  });
  elements.tooltipLayer.hidden = true;
  elements.tooltipLayer.textContent = "";
  elements.tooltipLayer.style.left = "";
  elements.tooltipLayer.style.width = "";
}

function positionTooltipLayer() {
  const openIcon = document.querySelector(".info-icon.is-open");
  if (!openIcon || !elements.settingsPanel) {
    return;
  }

  const panelRect = elements.settingsPanel.getBoundingClientRect();
  const viewportPadding = 12;
  const width = Math.min(panelRect.width, window.innerWidth - viewportPadding * 2);
  const left = Math.min(
    Math.max(viewportPadding, panelRect.left),
    window.innerWidth - viewportPadding - width
  );

  elements.tooltipLayer.style.left = `${left}px`;
  elements.tooltipLayer.style.width = `${width}px`;
}

function openTooltip(icon) {
  elements.tooltipLayer.textContent = icon.dataset.tooltip || "";
  elements.tooltipLayer.hidden = false;
  icon.classList.add("is-open");
  positionTooltipLayer();
}

function getRvmDownsampleRatio() {
  return Number(elements.rvmDetail.value) / 100;
}

function syncProcessedFps() {
  state.processedFps = Number(elements.processFps.value);
}

function updateButtons() {
  const hasVideo = Boolean(state.videoUrl);
  const hasProcessedFrames = state.processedFrames.length > 0;
  elements.startButton.disabled = !hasVideo || !state.videoReady || state.processingVideo || state.recording;
  elements.pauseButton.disabled = !hasProcessedFrames || state.processingVideo || state.recording;
  elements.pauseButton.textContent = state.running ? "Pause" : "Play";
  elements.loopButton.disabled = !hasProcessedFrames || state.processingVideo || state.recording;
  elements.loopButton.textContent = state.loopPlayback ? "Loop On" : "Loop Off";
  elements.loopButton.classList.toggle("is-on", state.loopPlayback);
  elements.cancelButton.disabled = !state.processingVideo;
  elements.exportButton.disabled = !hasProcessedFrames || state.recording || state.processingVideo;
  if (elements.engineMeta) {
    elements.engineMeta.textContent = getEngineLabel();
  }
  syncEngineAwareControls();
}

function showCanvasPreview() {
  elements.previewVideo.hidden = true;
  elements.outputCanvas.hidden = false;
}

function showVideoPreview() {
  elements.outputCanvas.hidden = true;
  elements.previewVideo.hidden = false;
}

function clearPreviewVideo() {
  elements.previewVideo.pause();
  elements.previewVideo.loop = false;
  elements.previewVideo.removeAttribute("src");
  elements.previewVideo.load();
  state.previewVideoUrl = "";
  state.previewVideoDuration = 0;
  elements.previewVideo.playbackRate = 1;
  updateTimingDebug();
  showCanvasPreview();
}

function updateUploadState(selected, detail = "") {
  elements.uploadCard.classList.toggle("is-selected", selected);
  if (selected) {
    elements.uploadTitle.textContent = detail || state.selectedFileName || "Video selected";
    elements.uploadDetail.textContent = "";
    return;
  }

  elements.uploadTitle.textContent = "Drop video here or click to browse";
  elements.uploadDetail.textContent = "";
}

async function ensureBackend() {
  if (state.backendReady) {
    return true;
  }

  const engine = getSelectedEngine();
  setStatus(`Connecting to ${getEngineLabel(engine)}...`);
  const response = await fetch(`/api/health?engine=${encodeURIComponent(engine)}`);
  if (!response.ok) {
    throw new Error(`${engine === "rvm" ? "RVM" : "Core ML"} worker is not reachable.`);
  }

  state.backendInfo = await response.json();
  state.backendReady = true;
  setStatus(engine === "rvm"
    ? `RVM ready on ${state.backendInfo.device || "unknown device"}.`
    : `${getEngineLabel(engine)} ready.`);
  updateButtons();
  return true;
}

async function ensureSession() {
  await ensureBackend();

  if (state.sessionId) {
    return state.sessionId;
  }

  const response = await fetch(`/api/session?engine=${encodeURIComponent(getSelectedEngine())}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ engine: getSelectedEngine() })
  });
  if (!response.ok) {
    throw new Error("Could not create a processing session.");
  }

  const payload = await response.json();
  state.sessionId = payload.session;
  return state.sessionId;
}

async function resetBackendSession() {
  const sessionId = await ensureSession();
  const response = await fetch(`/api/session/${encodeURIComponent(sessionId)}/reset?engine=${encodeURIComponent(getSelectedEngine())}`, {
    method: "POST"
  });

  if (!response.ok) {
    throw new Error("Could not reset the RVM session.");
  }

  resetTemporalMask();
}

async function closeBackendSession() {
  if (!state.sessionId) {
    return;
  }

  const sessionId = state.sessionId;
  state.sessionId = "";

  try {
    await fetch(`/api/session/${encodeURIComponent(sessionId)}?engine=${encodeURIComponent(getSelectedEngine())}`, {
      method: "DELETE"
    });
  } catch (error) {
    console.error(error);
  }
}

function resizeCanvases(width, height) {
  elements.outputCanvas.width = width;
  elements.outputCanvas.height = height;
  elements.sourceCanvas.width = width;
  elements.sourceCanvas.height = height;
  elements.maskCanvas.width = width;
  elements.maskCanvas.height = height;
  state.subjectCanvas = document.createElement("canvas");
  state.subjectCanvas.width = width;
  state.subjectCanvas.height = height;
  state.subjectContext = state.subjectCanvas.getContext("2d", { alpha: true });
  state.previousMaskData = null;
  elements.renderMeta.textContent = `${width} x ${height}`;
}

function probeVideoMetadata(fileUrl) {
  return new Promise((resolve, reject) => {
    const probe = document.createElement("video");
    const cleanup = () => {
      probe.removeAttribute("src");
      probe.load();
    };

    probe.preload = "metadata";
    probe.muted = true;
    probe.playsInline = true;

    probe.addEventListener(
      "loadedmetadata",
      () => {
        const metadata = {
          width: probe.videoWidth,
          height: probe.videoHeight,
          duration: probe.duration
        };
        cleanup();
        resolve(metadata);
      },
      { once: true }
    );

    probe.addEventListener(
      "error",
      () => {
        cleanup();
        reject(new Error("This video could not be decoded by the browser."));
      },
      { once: true }
    );

    probe.src = fileUrl;
    probe.load();
  });
}

async function loadVideo(file) {
  if (state.videoUrl) {
    URL.revokeObjectURL(state.videoUrl);
  }

  stopPreview();
  closeBackendSession();
  state.previousMaskData = null;
  state.latestFramePayload = null;
  state.videoReady = false;
  clearProcessedFrames();
  state.backendReady = false;
  state.backendInfo = null;
  state.sessionId = "";
  state.videoUrl = URL.createObjectURL(file);
  state.selectedFileName = file.name;
  updateUploadState(true, file.name);
  if (elements.videoMeta) {
    elements.videoMeta.textContent = `${file.name} selected`;
  }
  if (elements.playbackMeta) {
    elements.playbackMeta.textContent = "Loading video metadata...";
  }
  setStatus("Video selected. Loading metadata...");
  updateButtons();

  try {
    const metadata = await probeVideoMetadata(state.videoUrl);
    syncProcessedFps();
    resizeCanvases(metadata.width, metadata.height);
    state.processedDuration = metadata.duration;
    state.videoReady = true;
    if (elements.videoMeta) {
      elements.videoMeta.textContent = `${metadata.width} x ${metadata.height} • ${metadata.duration.toFixed(2)}s`;
    }
    if (elements.playbackMeta) {
      elements.playbackMeta.textContent = "Ready to process.";
    }
    setStatus("Video ready. Click Process video.");
    updateUploadState(true, `${file.name} • ${metadata.width}×${metadata.height} • ${metadata.duration.toFixed(2)}s`);

    elements.sourceVideo.src = state.videoUrl;
    elements.sourceVideo.load();
  } catch (error) {
    console.error(error);
    state.videoReady = false;
    state.selectedFileName = "";
    updateUploadState(false);
    if (elements.playbackMeta) {
      elements.playbackMeta.textContent = "Video load failed.";
    }
    setStatus(error.message);
    elements.uploadDetail.textContent = "This video could not be loaded by the browser.";
  }

  updateButtons();
}

async function handleVideoReady() {
  if (!state.videoReady && elements.sourceVideo.videoWidth && elements.sourceVideo.videoHeight) {
    syncProcessedFps();
    resizeCanvases(elements.sourceVideo.videoWidth, elements.sourceVideo.videoHeight);
    state.processedDuration = elements.sourceVideo.duration;
    state.videoReady = true;
    if (elements.videoMeta) {
      elements.videoMeta.textContent = `${elements.sourceVideo.videoWidth} x ${elements.sourceVideo.videoHeight} • ${elements.sourceVideo.duration.toFixed(2)}s`;
    }
    if (elements.playbackMeta) {
      elements.playbackMeta.textContent = "Ready to process.";
    }
    setStatus("Video ready. Click Process video.");
  }
  updateButtons();
}

function applyForegroundTuning() {
  const boost = Number(elements.foregroundBoost.value) / 100;
  if (boost === 1) {
    return;
  }

  const imageData = outputContext.getImageData(0, 0, elements.outputCanvas.width, elements.outputCanvas.height);
  const data = imageData.data;

  for (let index = 0; index < data.length; index += 4) {
    data[index] = Math.min(255, data[index] * boost);
    data[index + 1] = Math.min(255, data[index + 1] * boost);
    data[index + 2] = Math.min(255, data[index + 2] * boost);
  }

  outputContext.putImageData(imageData, 0, 0);
}

function drawBackground() {
  const width = elements.outputCanvas.width;
  const height = elements.outputCanvas.height;

  outputContext.clearRect(0, 0, width, height);

  if (elements.backgroundMode.value === "transparent") {
    return;
  }

  if (elements.backgroundMode.value === "image" && state.bgImage) {
    const sourceRatio = state.bgImage.width / state.bgImage.height;
    const targetRatio = width / height;
    let drawWidth = width;
    let drawHeight = height;
    let offsetX = 0;
    let offsetY = 0;

    if (sourceRatio > targetRatio) {
      drawHeight = height;
      drawWidth = height * sourceRatio;
      offsetX = (width - drawWidth) / 2;
    } else {
      drawWidth = width;
      drawHeight = width / sourceRatio;
      offsetY = (height - drawHeight) / 2;
    }

    outputContext.drawImage(state.bgImage, offsetX, offsetY, drawWidth, drawHeight);
    return;
  }

  outputContext.fillStyle = elements.backgroundColor.value;
  outputContext.fillRect(0, 0, width, height);
}

function buildAlphaMask(maskSource) {
  const width = elements.maskCanvas.width;
  const height = elements.maskCanvas.height;
  const feather = Number(elements.feather.value);
  const threshold = Number(elements.threshold.value) / 100;
  const maskStrength = Number(elements.maskStrength.value) / 100;
  const temporalSmooth = Number(elements.temporalSmooth.value) / 100;
  const edgeShrink = Number(elements.edgeShrink.value);

  maskContext.save();
  maskContext.clearRect(0, 0, width, height);
  const blurValue = feather ? `blur(${feather}px)` : "none";
  const scale = edgeShrink ? 1 - edgeShrink / Math.max(width, height) : 1;
  maskContext.filter = blurValue;
  if (edgeShrink) {
    maskContext.translate(width / 2, height / 2);
    maskContext.scale(scale, scale);
    maskContext.translate(-width / 2, -height / 2);
  }
  maskContext.drawImage(maskSource, 0, 0, width, height);
  maskContext.restore();

  const imageData = maskContext.getImageData(0, 0, width, height);
  const data = imageData.data;
  const nextMask = new Uint8ClampedArray(width * height);
  const previousMask = state.previousMaskData;
  let nonZeroPixels = 0;
  let maxAlpha = 0;

  for (let index = 0; index < data.length; index += 4) {
    const pixelIndex = index / 4;
    const confidence = Math.max(0, Math.min(1, (data[index] / 255) * maskStrength));
    let alpha = (confidence - threshold) / Math.max(0.001, 1 - threshold);
    alpha = Math.max(0, Math.min(1, alpha));

    if (previousMask) {
      const previousAlpha = previousMask[pixelIndex] / 255;
      alpha = previousAlpha * temporalSmooth + alpha * (1 - temporalSmooth);
    }

    const value = Math.round(alpha * 255);
    if (value > 0) {
      nonZeroPixels += 1;
    }
    if (value > maxAlpha) {
      maxAlpha = value;
    }
    nextMask[pixelIndex] = value;
    data[index] = value;
    data[index + 1] = value;
    data[index + 2] = value;
    data[index + 3] = 255;
  }

  state.previousMaskData = nextMask;
  maskContext.putImageData(imageData, 0, 0);
  return {
    coverage: nonZeroPixels / (width * height),
    maxAlpha
  };
}

function renderCompositedFrame(maskSource) {
  const width = elements.outputCanvas.width;
  const height = elements.outputCanvas.height;
  const shadow = Number(elements.shadow.value);

  const maskStats = buildAlphaMask(maskSource);
  drawBackground();

  if (maskStats.maxAlpha < 8 || maskStats.coverage < 0.0015) {
    outputContext.drawImage(elements.sourceCanvas, 0, 0, width, height);
    setStatus("No clear subject found in this frame yet. Try lowering threshold or using footage with a person in view.");
    return;
  }

  state.latestMaskStats = maskStats;

  outputContext.save();
  if (shadow > 0) {
    outputContext.shadowColor = "rgba(0, 0, 0, 0.18)";
    outputContext.shadowBlur = shadow;
  }
  outputContext.drawImage(sourceCanvasWithMask(), 0, 0, width, height);
  outputContext.restore();

  applyForegroundTuning();
}

function sourceCanvasWithMask(source = elements.sourceCanvas) {
  const width = elements.sourceCanvas.width;
  const height = elements.sourceCanvas.height;
  const ctx = state.subjectContext;

  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(source, 0, 0, width, height);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(elements.maskCanvas, 0, 0, width, height);
  ctx.globalCompositeOperation = "source-over";
  return state.subjectCanvas;
}

function drawCutout(cutoutImage, maskSource) {
  const width = elements.outputCanvas.width;
  const height = elements.outputCanvas.height;
  const shadow = Number(elements.shadow.value);
  const maskStats = buildAlphaMask(maskSource);

  drawBackground();

  if (maskStats.maxAlpha < 8 || maskStats.coverage < 0.0015) {
    outputContext.drawImage(elements.sourceCanvas, 0, 0, width, height);
    setStatus("No clear subject found in this frame yet. Try lowering threshold or using footage with a person in view.");
    return;
  }

  state.latestMaskStats = maskStats;
  const maskedCutout = sourceCanvasWithMask(cutoutImage);

  if (shadow > 0) {
    outputContext.save();
    outputContext.shadowColor = "rgba(0, 0, 0, 0.18)";
    outputContext.shadowBlur = shadow;
    outputContext.drawImage(maskedCutout, 0, 0, width, height);
    outputContext.restore();
  } else {
    outputContext.drawImage(maskedCutout, 0, 0, width, height);
  }

  applyForegroundTuning();
}

async function renderPayload(payload) {
  if (!payload) {
    return;
  }

  const maskBlob = await fetch(`data:image/png;base64,${payload.mask}`).then((result) => result.blob());
  const cutoutBlob = await fetch(`data:image/png;base64,${payload.cutout}`).then((result) => result.blob());
  const maskBitmap = await createImageBitmap(maskBlob);
  const cutoutBitmap = await createImageBitmap(cutoutBlob);
  drawCutout(cutoutBitmap, maskBitmap);
  maskBitmap.close();
  cutoutBitmap.close();
}

function adaptPreviewPayload(payload) {
  if (!payload) {
    return null;
  }

  if (payload.mask && payload.cutout) {
    return payload;
  }

  const alphaImage = payload.alphaImage || "";
  const cutoutImage = payload.cutoutImage || "";
  if (!alphaImage || !cutoutImage) {
    return null;
  }

  return {
    mask: alphaImage.replace(/^data:image\/png;base64,/, ""),
    cutout: cutoutImage.replace(/^data:image\/png;base64,/, "")
  };
}

function getProgressLabel(phase, engine, progress, processedFrames, frameCount) {
  const engineLabel = getEngineLabel(engine);
  if (phase === "extracting_frames") {
    return `PREPARING VIDEO FOR ${engineLabel.toUpperCase()}: ${progress}%`;
  }
  if (phase === "processing_frames") {
    if (frameCount) {
      return `PROCESSING VIDEO: ${progress}% (${processedFrames}/${frameCount})`;
    }
    return `PROCESSING VIDEO: ${progress}%`;
  }
  if (phase === "encoding_preview") {
    return `CONVERTING FRAMES: ${progress}%`;
  }
  return `PROCESSING VIDEO: ${progress}%`;
}

async function rerenderCurrentFrame() {
  const frame = state.processedFrames[state.processedFrameIndex];
  if (frame) {
    await ensureDecodedFrame(frame);
    drawCutout(frame.cutoutBitmap, frame.maskBitmap);
    return;
  }

  if (!state.latestFramePayload) {
    return;
  }

  try {
    await renderPayload(state.latestFramePayload);
  } catch (error) {
    console.error(error);
  }
}

async function processCurrentFrame({ renderPreview = true } = {}) {
  if (!state.backendReady || state.processingFrame) {
    return;
  }

  state.processingFrame = true;
  const sessionId = await ensureSession();

  const width = elements.sourceCanvas.width;
  const height = elements.sourceCanvas.height;
  sourceContext.clearRect(0, 0, width, height);
  sourceContext.drawImage(elements.sourceVideo, 0, 0, width, height);

  const blob = await new Promise((resolve) => {
    elements.sourceCanvas.toBlob(resolve, "image/jpeg", 0.9);
  });

  if (!blob) {
    state.processingFrame = false;
    throw new Error("Failed to encode frame.");
  }

  const response = await fetch(
    `/api/session/${encodeURIComponent(sessionId)}/matte?engine=${encodeURIComponent(getSelectedEngine())}&downsampleRatio=${encodeURIComponent(getRvmDownsampleRatio())}`,
    {
    method: "POST",
    headers: {
      "Content-Type": "image/jpeg"
    },
    body: blob
    }
  );

  if (!response.ok) {
    state.processingFrame = false;
    let message = "Matte request failed.";
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch (error) {
      // Leave the fallback message in place.
    }
    throw new Error(message);
  }

  const payload = await response.json();
  state.latestFramePayload = payload;
  if (renderPreview) {
    await renderPayload(payload);
  }
  state.processingFrame = false;
  return payload;
}

function clearProcessedFrames() {
  clearPreviewVideo();
  state.processedFrames.forEach((frame) => {
    if (frame.maskBitmap) {
      frame.maskBitmap.close();
    }
    if (frame.cutoutBitmap) {
      frame.cutoutBitmap.close();
    }
    if (frame.renderedBitmap) {
      frame.renderedBitmap.close();
    }
  });
  state.processedFrames = [];
  state.processedFrameIndex = 0;
  state.processedFramesDirty = false;
  setFinishedMeta("Waiting for run");
  updateTimingDebug();
  updateButtons();
}

function seekVideo(targetTime) {
  return new Promise((resolve, reject) => {
    const onSeeked = () => {
      cleanup();
      resolve();
    };

    const onError = () => {
      cleanup();
      reject(new Error("Video seek failed."));
    };

    const cleanup = () => {
      elements.sourceVideo.removeEventListener("seeked", onSeeked);
      elements.sourceVideo.removeEventListener("error", onError);
    };

    elements.sourceVideo.addEventListener("seeked", onSeeked, { once: true });
    elements.sourceVideo.addEventListener("error", onError, { once: true });
    elements.sourceVideo.currentTime = Math.max(0, Math.min(targetTime, Math.max(0, elements.sourceVideo.duration - seekEpsilon)));
  });
}

async function decodeFramePayload(payload) {
  const maskBlob = await fetch(`data:image/png;base64,${payload.mask}`).then((result) => result.blob());
  const cutoutBlob = await fetch(`data:image/png;base64,${payload.cutout}`).then((result) => result.blob());
  const maskBitmap = await createImageBitmap(maskBlob);
  const cutoutBitmap = await createImageBitmap(cutoutBlob);
  return {
    maskBitmap,
    cutoutBitmap
  };
}

async function ensureDecodedFrame(frame) {
  if (frame.maskBitmap && frame.cutoutBitmap) {
    return frame;
  }

  const decoded = await decodeFramePayload(frame.payload);
  frame.maskBitmap = decoded.maskBitmap;
  frame.cutoutBitmap = decoded.cutoutBitmap;
  return frame;
}

async function renderFrameToBitmap(frame) {
  await ensureDecodedFrame(frame);
  drawCutout(frame.cutoutBitmap, frame.maskBitmap);
  if (frame.renderedBitmap) {
    frame.renderedBitmap.close();
  }
  frame.renderedBitmap = await createImageBitmap(elements.outputCanvas);
}

async function drawFrameToOutput(frame) {
  if (frame.renderedBitmap) {
    outputContext.clearRect(0, 0, elements.outputCanvas.width, elements.outputCanvas.height);
    outputContext.drawImage(frame.renderedBitmap, 0, 0, elements.outputCanvas.width, elements.outputCanvas.height);
    return;
  }

  await ensureDecodedFrame(frame);
  drawCutout(frame.cutoutBitmap, frame.maskBitmap);
}

function getPreviewEncodeSize() {
  const width = elements.outputCanvas.width;
  const height = elements.outputCanvas.height;
  const maxDimension = 960;
  const longestEdge = Math.max(width, height);

  if (longestEdge <= maxDimension) {
    return { width, height };
  }

  const scale = maxDimension / longestEdge;
  return {
    width: Math.max(2, Math.round(width * scale)),
    height: Math.max(2, Math.round(height * scale))
  };
}

function encodeCanvasBlob({ purpose }) {
  if (purpose !== "preview") {
    return new Promise((resolve) => {
      elements.outputCanvas.toBlob(resolve, "image/png");
    });
  }

  const { width, height } = getPreviewEncodeSize();
  const previewCanvas = document.createElement("canvas");
  previewCanvas.width = width;
  previewCanvas.height = height;
  const previewContext = previewCanvas.getContext("2d", { alpha: true });
  previewContext.drawImage(elements.outputCanvas, 0, 0, width, height);
  return new Promise((resolve) => {
    previewCanvas.toBlob(resolve, "image/png");
  });
}

async function prepareRenderedFrames() {
  if (!state.processedFrames.length) {
    return;
  }

  const needsRender = state.processedFramesDirty || state.processedFrames.some((frame) => !frame.renderedBitmap);
  if (!needsRender) {
    return;
  }

    setStatus("REFRESHING RENDERED FRAMES");
  for (let index = 0; index < state.processedFrames.length; index += 1) {
    if (state.processedFramesDirty || !state.processedFrames[index].renderedBitmap) {
      await renderFrameToBitmap(state.processedFrames[index]);
    }
  }
  state.processedFramesDirty = false;
  setStatus("RENDERED FRAMES REFRESHED");
}

async function processVideo() {
  await processVideoWithBatch();
}

async function processVideoWithBatch() {
  const engine = getSelectedEngine();
  stopPreview();
  clearProcessedFrames();
  syncProcessedFps();

  state.processingVideo = true;
  state.cancelProcessing = false;
  state.latestFramePayload = null;
  state.processingStartedAt = Date.now();
  setFinishedMeta("Processing");
  updateButtons();

  try {
    setStatus(`UPLOADING VIDEO TO ${getEngineLabel(engine).toUpperCase()}`);
    if (elements.playbackMeta) {
      elements.playbackMeta.textContent = "PROCESSING VIDEO";
    }
    const videoBlob = await fetch(state.videoUrl).then((result) => result.blob());

    const response = await fetch("/api/process-video/start", {
      method: "POST",
      headers: {
        "Content-Type": videoBlob.type || "application/octet-stream",
        "X-File-Name": elements.uploadTitle.textContent || "backdrop-preview.mp4",
        "X-Engine": engine,
        "X-Process-Fps": String(state.processedFps),
        "X-Rvm-Detail": String(getRvmDownsampleRatio())
      },
      body: videoBlob
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || `${engine === "rvm" ? "RVM" : "Core ML"} batch processing failed.`);
    }

    const jobId = result.jobId;
    let jobResult = null;
    while (!jobResult) {
      if (state.cancelProcessing) {
        throw new Error("Processing cancelled.");
      }

      await new Promise((resolve) => setTimeout(resolve, 250));
      const statusResponse = await fetch(`/api/process-video/status?jobId=${encodeURIComponent(jobId)}`);
      const statusResult = await statusResponse.json();
      if (!statusResponse.ok) {
        throw new Error(statusResult.error || "Could not read processing status.");
      }

      const progress = Math.max(0, Math.min(100, Math.round(Number(statusResult.progress) || 0)));
      setStatus(getProgressLabel(
        statusResult.phase,
        engine,
        progress,
        statusResult.processedFrames,
        statusResult.frameCount
      ));

      if (statusResult.status === "failed") {
        throw new Error(statusResult.error || "Video processing failed.");
      }

      if (statusResult.status === "completed") {
        jobResult = statusResult.result;
      }
    }

    state.previewVideoUrl = `${jobResult.outputUrl}?t=${Date.now()}`;
    state.previewVideoDuration = Number(jobResult.outputMeta?.duration) || 0;
    state.processedDuration = Number(jobResult.outputMeta?.duration) || state.processedDuration;
    state.processedFrames = new Array(Number(jobResult.frameCount) || 0).fill(null).map(() => ({
      payload: null,
      maskBitmap: null,
      cutoutBitmap: null,
      renderedBitmap: null
    }));
    state.processedFrameIndex = 0;
    state.processedFramesDirty = false;
    updateTimingDebug();

    const previewPayload = adaptPreviewPayload(jobResult.preview);
    if (previewPayload) {
      state.latestFramePayload = previewPayload;
      await renderPayload(previewPayload);
      showCanvasPreview();
    }

    await loadPreviewVideo(state.previewVideoUrl);
    const workerSeconds = Number(jobResult.workerElapsedSeconds) || 0;
    if (elements.playbackMeta) {
      elements.playbackMeta.textContent = workerSeconds > 0
        ? `CAPTURED ${state.processedFrames.length} / ${state.processedFrames.length} FRAMES • ${getEngineLabel(engine).toUpperCase()} WORKER ${workerSeconds.toFixed(1)}S`
        : `CAPTURED ${state.processedFrames.length} / ${state.processedFrames.length} FRAMES`;
    }
    setStatus("Completed");
    setFinishedMeta(`Finished in ${formatElapsed(Date.now() - state.processingStartedAt)}`);
  } finally {
    state.processingVideo = false;
    state.cancelProcessing = false;
    updateButtons();
  }
}

function renderProcessedFrame(index) {
  const frame = state.processedFrames[index];
  if (!frame) {
    return;
  }

  state.processedFrameIndex = index;
  if (frame.renderedBitmap) {
    outputContext.clearRect(0, 0, elements.outputCanvas.width, elements.outputCanvas.height);
    outputContext.drawImage(frame.renderedBitmap, 0, 0, elements.outputCanvas.width, elements.outputCanvas.height);
  } else {
    drawCutout(frame.cutoutBitmap, frame.maskBitmap);
  }
  const currentTime = index / state.processedFps;
  if (elements.playbackMeta) {
    elements.playbackMeta.textContent = `${currentTime.toFixed(2)}s / ${state.processedDuration.toFixed(2)}s`;
  }
}

async function encodeProcessedVideo({ purpose = "preview" } = {}) {
  const hasFramePayloads = state.processedFrames.some((frame) => frame?.payload);
  if (!hasFramePayloads && state.previewVideoUrl) {
    return {
      ok: true,
      filename: "backdrop-export.mp4",
      downloadUrl: state.previewVideoUrl.replace(/\?.*$/, ""),
      duration: state.previewVideoDuration || state.processedDuration || 0,
    };
  }

  stopPreview();
  syncProcessedFps();
  state.recording = true;
  updateButtons();
  setStatus(`CONVERTING FRAMES TO ${purpose.toUpperCase()} MP4`);

  const dimensions = purpose === "preview"
    ? getPreviewEncodeSize()
    : { width: elements.outputCanvas.width, height: elements.outputCanvas.height };

  try {
    const startResponse = await fetch("/api/export/start", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        purpose,
        frameCount: state.processedFrames.length,
        fps: state.processedFps,
        width: dimensions.width,
        height: dimensions.height
      })
    });

    if (!startResponse.ok) {
      const payload = await startResponse.json().catch(() => ({}));
      throw new Error(payload.error || `Could not start ${purpose} export.`);
    }

    const { exportId } = await startResponse.json();

    for (let index = 0; index < state.processedFrames.length; index += 1) {
      state.processedFrameIndex = index;
      await drawFrameToOutput(state.processedFrames[index]);
      const blob = await encodeCanvasBlob({ purpose });
      if (!blob) {
        throw new Error(`Failed to encode ${purpose} frame ${index + 1}.`);
      }

      const frameResponse = await fetch(`/api/export/${encodeURIComponent(exportId)}/frame?index=${index}`, {
        method: "POST",
        headers: {
          "Content-Type": "image/png"
        },
        body: blob
      });

      if (!frameResponse.ok) {
        const payload = await frameResponse.json().catch(() => ({}));
        throw new Error(payload.error || `Could not upload ${purpose} frame ${index + 1}.`);
      }

      const percent = Math.round(((index + 1) / state.processedFrames.length) * 100);
      setStatus(`UPLOADING ${purpose.toUpperCase()} FRAMES: ${percent}%`);
      if (elements.playbackMeta) {
        elements.playbackMeta.textContent = `CAPTURED ${index + 1} / ${state.processedFrames.length} FRAMES`;
      }
    }

    const finalizeResponse = await fetch(`/api/export/${encodeURIComponent(exportId)}/finalize`, {
      method: "POST"
    });

    if (!finalizeResponse.ok) {
      const payload = await finalizeResponse.json().catch(() => ({}));
      throw new Error(payload.error || `ffmpeg ${purpose} export failed.`);
    }

    return finalizeResponse.json();
  } finally {
    state.recording = false;
    updateButtons();
  }
}

function loadPreviewVideo(url) {
  return new Promise((resolve, reject) => {
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("The encoded preview video could not be loaded."));
    };
    const cleanup = () => {
      elements.previewVideo.removeEventListener("loadeddata", onLoaded);
      elements.previewVideo.removeEventListener("error", onError);
    };

    elements.previewVideo.addEventListener("loadeddata", onLoaded, { once: true });
    elements.previewVideo.addEventListener("error", onError, { once: true });
    elements.previewVideo.src = url;
    elements.previewVideo.load();
  });
}

async function startPlayback() {
  if (!state.processedFrames.length) {
    return;
  }

  if (!state.previewVideoUrl) {
    const { downloadUrl, duration } = await encodeProcessedVideo({ purpose: "preview" });
    state.previewVideoUrl = `${downloadUrl}?t=${Date.now()}`;
    state.previewVideoDuration = Number(duration) || 0;
    updateTimingDebug();
    await loadPreviewVideo(state.previewVideoUrl);
  }

  state.running = true;
  elements.previewVideo.currentTime = Math.min(
    state.processedDuration,
    state.processedFrameIndex / state.processedFps
  );
  const expectedDuration = state.processedFrames.length / state.processedFps;
  const actualDuration = state.previewVideoDuration || elements.previewVideo.duration || expectedDuration;
  const playbackRate = actualDuration > 0 ? Math.min(4, Math.max(0.25, actualDuration / expectedDuration)) : 1;
  elements.previewVideo.playbackRate = playbackRate;
  elements.previewVideo.loop = state.loopPlayback;
  updateTimingDebug();
  showVideoPreview();
  updateButtons();
  await elements.previewVideo.play();
}

function stopPreview() {
  if (elements.previewVideo.currentSrc) {
    state.processedFrameIndex = Math.max(0, Math.floor(elements.previewVideo.currentTime * state.processedFps));
    elements.previewVideo.pause();
  }
  state.running = false;
  updateButtons();
}

function resetTemporalMask() {
  state.previousMaskData = null;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

async function exportVideo() {
  if (!state.processedFrames.length) {
    throw new Error("Process the video before exporting.");
  }

  const { downloadUrl, filename } = await encodeProcessedVideo({ purpose: "export" });
  const videoResponse = await fetch(downloadUrl);
  if (!videoResponse.ok) {
    throw new Error("Could not download exported MP4.");
  }
  const blob = await videoResponse.blob();
  downloadBlob(blob, filename || "backdrop-export.mp4");

  state.recording = false;
  updateButtons();
  setStatus("Export finished.");
  if (elements.playbackMeta) {
    elements.playbackMeta.textContent = "Export complete.";
  }
}

elements.videoInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }

  loadVideo(file).catch((error) => {
    console.error(error);
    setStatus(error.message);
  });
});

["dragenter", "dragover"].forEach((eventName) => {
  document.addEventListener(eventName, (event) => {
    event.preventDefault();
  });
});

["dragleave", "drop"].forEach((eventName) => {
  document.addEventListener(eventName, (event) => {
    event.preventDefault();
  });
});

document.querySelector(".upload").addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file && file.type.startsWith("video/")) {
    elements.videoInput.files = event.dataTransfer.files;
    loadVideo(file).catch((error) => {
      console.error(error);
      setStatus(error.message);
    });
  }
});

document.querySelectorAll(".info-icon").forEach((icon) => {
  icon.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const shouldOpen = !icon.classList.contains("is-open");
    closeTooltips();
    if (shouldOpen) {
      openTooltip(icon);
    }
  });
});

document.addEventListener("click", () => {
  closeTooltips();
});

window.addEventListener("resize", positionTooltipLayer);
window.addEventListener("scroll", positionTooltipLayer, { passive: true });

elements.backgroundImage.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) {
    state.bgImage = null;
    return;
  }

  const image = new Image();
  image.src = URL.createObjectURL(file);
  await image.decode();
  state.bgImage = image;
  setStatus("Background image loaded.");
  rerenderCurrentFrame();
});

elements.sourceVideo.addEventListener("loadedmetadata", () => {
  handleVideoReady().catch((error) => {
    console.error(error);
    setStatus(error.message);
  });
});
elements.sourceVideo.addEventListener("loadeddata", () => {
  if (!state.backendReady && elements.sourceVideo.videoWidth && elements.sourceVideo.videoHeight) {
    handleVideoReady().catch((error) => {
      console.error(error);
      setStatus(error.message);
    });
  }
});
elements.sourceVideo.addEventListener("error", () => {
  setStatus("This video could not be loaded. Try another format or file.");
  if (elements.playbackMeta) {
    elements.playbackMeta.textContent = "Video load failed.";
  }
});
elements.previewVideo.addEventListener("pause", () => {
  if (!state.running) {
    return;
  }

  state.running = false;
  state.processedFrameIndex = Math.max(0, Math.floor(elements.previewVideo.currentTime * state.processedFps));
  updateTimingDebug();
  updateButtons();
});
elements.previewVideo.addEventListener("ended", () => {
  state.running = false;
  state.processedFrameIndex = 0;
  elements.previewVideo.currentTime = 0;
  updateTimingDebug();
  updateButtons();
});
elements.startButton.addEventListener("click", async () => {
  try {
    await processVideo();
  } catch (error) {
    console.error(error);
    setStatus(error.message);
    state.processingVideo = false;
    updateButtons();
  }
});
elements.pauseButton.addEventListener("click", () => {
  if (state.running) {
    stopPreview();
    return;
  }

  startPlayback().catch((error) => {
    console.error(error);
    setStatus(error.message);
  });
});
elements.loopButton.addEventListener("click", () => {
  state.loopPlayback = !state.loopPlayback;
  elements.previewVideo.loop = state.loopPlayback;
  updateButtons();
});
elements.cancelButton.addEventListener("click", () => {
  if (!state.processingVideo) {
    return;
  }

  state.cancelProcessing = true;
  setStatus("Cancelling after the current frame...");
  updateButtons();
});
elements.exportButton.addEventListener("click", async () => {
  try {
    await exportVideo();
  } catch (error) {
    console.error(error);
    setStatus(error.message);
    state.recording = false;
    updateButtons();
  }
});

[
  elements.backgroundMode,
  elements.backgroundColor,
  elements.threshold,
  elements.feather,
  elements.maskStrength,
  elements.temporalSmooth,
  elements.edgeShrink,
  elements.foregroundBoost,
  elements.shadow
].forEach((input) => {
  input.addEventListener("input", () => {
    resetTemporalMask();
    clearPreviewVideo();
    if (!elements.sourceVideo.src) {
      return;
    }

    setStatus("Settings updated.");
    if (state.processedFrames.length) {
      state.processedFramesDirty = true;
    }
    rerenderCurrentFrame();
  });
});

elements.processFps.addEventListener("input", () => {
  syncProcessedFps();
  clearProcessedFrames();
  if (elements.sourceVideo.src) {
    setStatus(`Process FPS changed to ${state.processedFps}. Process video again.`);
    if (elements.playbackMeta) {
      elements.playbackMeta.textContent = "Process FPS changed. Reprocess to preview.";
    }
  }
});

elements.rvmDetail.addEventListener("input", async () => {
  resetTemporalMask();
  state.latestFramePayload = null;
  clearProcessedFrames();
  setStatus(`RVM detail updated to ${getRvmDownsampleRatio().toFixed(2)}x.`);

  if (!elements.sourceVideo.src) {
    return;
  }

  if (elements.playbackMeta) {
    elements.playbackMeta.textContent = "RVM detail changed. Process video again.";
  }
});

elements.engineSelect.addEventListener("change", async () => {
  stopPreview();
  await closeBackendSession();
  state.backendReady = false;
  state.backendInfo = null;
  state.sessionId = "";
  state.latestFramePayload = null;
  resetTemporalMask();
  clearProcessedFrames();
  setStatus(`${getEngineLabel()} selected. Process video again.`);
  if (elements.playbackMeta) {
    elements.playbackMeta.textContent = "Engine changed. Process video again.";
  }
  setFinishedMeta("Waiting for run");
  updateButtons();
});

window.addEventListener("beforeunload", () => {
  closeBackendSession();
});

updateTimingDebug();
updateUploadState(false);
setFinishedMeta("Waiting for run");
syncEngineAwareControls();
