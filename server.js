const http = require("http");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");
const { randomUUID } = require("crypto");

const host = process.env.HOST || "127.0.0.1";
const port = Number(process.env.PORT || 3010);
const root = path.join(__dirname, "public");
const appRoot = __dirname;
const labRoot = path.join(__dirname, "experiments", "coreml_lab");
const workRoot = path.join(os.tmpdir(), "backdrop");
const outputRoot = path.join(workRoot, "outputs");
const maxJsonBytes = 20 * 1024 * 1024;
const maxUploadBytes = 3 * 1024 * 1024 * 1024;
const maxImageBytes = 25 * 1024 * 1024;
const rvmBatchMaxDimension = 1463;
const maxVideoDurationSeconds = 2 * 60 * 60;
function buildScaleFilter(maxDimension) {
  return `scale=w='trunc(min(${maxDimension},iw)/2)*2':h='trunc((trunc(min(${maxDimension},iw)/2)*2)/a/2)*2'`;
}

function cleanupWorkRootContents() {
  if (!fs.existsSync(workRoot)) {
    return;
  }

  fs.readdirSync(workRoot).forEach((entry) => {
    fs.rmSync(path.join(workRoot, entry), {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 120,
    });
  });
}

cleanupWorkRootContents();
fs.mkdirSync(outputRoot, { recursive: true });
const exportJobs = new Map();
const engineSessions = new Map();
const videoJobs = new Map();

const bundledRvmPythonPaths = [
  path.join(appRoot, ".venv314", "bin", "python"),
  path.join(appRoot, ".venv", "bin", "python"),
];

function pickExecutable(candidates, fallback = "python3") {
  return candidates.find((candidate) => fs.existsSync(candidate)) || fallback;
}

function createWorkerController(name, executable, scriptPath, cwd, env = {}) {
  return {
    name,
    executable,
    scriptPath,
    cwd,
    env,
    process: null,
    buffer: "",
    counter: 0,
    pending: new Map(),
  };
}

const coremlWorker = createWorkerController(
  "coreml",
  path.join(labRoot, ".venv312", "bin", "python"),
  path.join(labRoot, "coreml_worker.py"),
  labRoot,
  { TMPDIR: path.join(labRoot, "tmp") }
);

const ppmattingWorker = createWorkerController(
  "ppmattingv2",
  path.join(__dirname, ".venv312", "bin", "python"),
  path.join(appRoot, "scripts", "ppmattingv2_worker.py"),
  appRoot,
  {
    TMPDIR: process.env.TMPDIR || "/tmp",
    TMP: process.env.TMP || "/tmp",
    TEMP: process.env.TEMP || "/tmp",
  }
);

const modnetWorker = createWorkerController(
  "modnet",
  pickExecutable(bundledRvmPythonPaths),
  path.join(appRoot, "scripts", "modnet_worker.py"),
  appRoot,
  {
    TMPDIR: process.env.TMPDIR || "/tmp",
    TMP: process.env.TMP || "/tmp",
    TEMP: process.env.TEMP || "/tmp",
    BGREMOVER_FORCE_CPU: process.env.BGREMOVER_FORCE_CPU || "0",
  }
);

const rvmWorker = createWorkerController(
  "rvm",
  pickExecutable(bundledRvmPythonPaths),
  path.join(appRoot, "scripts", "rvm_worker.py"),
  appRoot,
  {
    TMPDIR: process.env.TMPDIR || "/tmp",
    TMP: process.env.TMP || "/tmp",
    TEMP: process.env.TEMP || "/tmp",
    BGREMOVER_FORCE_CPU: process.env.BGREMOVER_FORCE_CPU || "0",
  }
);

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".mp4": "video/mp4",
};

function send(res, statusCode, body, contentType = "text/plain; charset=utf-8") {
  res.writeHead(statusCode, { "Content-Type": contentType });
  res.end(body);
}

function sendJson(res, statusCode, payload) {
  send(res, statusCode, JSON.stringify(payload), "application/json; charset=utf-8");
}

function sendFile(filePath, res) {
  fs.readFile(filePath, (error, data) => {
    if (error) {
      send(res, 404, "Not found");
      return;
    }
    send(res, 200, data, mimeTypes[path.extname(filePath).toLowerCase()] || "application/octet-stream");
  });
}

function readRequestBody(req, maxBytes) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let totalBytes = 0;

    req.on("data", (chunk) => {
      totalBytes += chunk.length;
      if (totalBytes > maxBytes) {
        reject(new Error("Video file is too large. The current upload limit is about 2 GB."));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });

    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function streamRequestToFile(req, destinationPath, maxBytes) {
  return new Promise((resolve, reject) => {
    const output = fs.createWriteStream(destinationPath);
    let totalBytes = 0;
    let settled = false;

    const finish = (error = null, bytesWritten = totalBytes) => {
      if (settled) {
        return;
      }
      settled = true;
      output.destroy();
      if (error) {
        fs.rmSync(destinationPath, { force: true });
        reject(error);
      } else {
        resolve({ bytesWritten });
      }
    };

    req.on("data", (chunk) => {
      totalBytes += chunk.length;
      if (totalBytes > maxBytes) {
        finish(new Error("Video file is too large. The current upload limit is 3 GB."));
        req.destroy();
        return;
      }
      if (!output.write(chunk)) {
        req.pause();
      }
    });

    output.on("drain", () => {
      req.resume();
    });

    req.on("end", () => {
      output.end(() => {
        if (!totalBytes) {
          finish(new Error("No video payload received."));
          return;
        }
        finish(null, totalBytes);
      });
    });

    req.on("error", (error) => finish(error));
    output.on("error", (error) => finish(error));
  });
}

function readJsonBody(req) {
  return readRequestBody(req, maxJsonBytes).then((buffer) => {
    if (!buffer.length) {
      return {};
    }
    return JSON.parse(buffer.toString("utf8"));
  });
}

function resetPending(controller, error) {
  for (const request of controller.pending.values()) {
    request.reject(error);
  }
  controller.pending.clear();
}

function startWorker(controller) {
  if (controller.process) {
    return controller.process;
  }

  controller.process = spawn(controller.executable, [controller.scriptPath], {
    cwd: controller.cwd,
    stdio: ["pipe", "pipe", "pipe"],
    env: {
      ...process.env,
      ...controller.env,
    },
  });

  controller.process.stdout.setEncoding("utf8");
  controller.process.stdout.on("data", (chunk) => {
    controller.buffer += chunk;
    let newlineIndex = controller.buffer.indexOf("\n");

    while (newlineIndex >= 0) {
      const line = controller.buffer.slice(0, newlineIndex).trim();
      controller.buffer = controller.buffer.slice(newlineIndex + 1);

      if (line) {
        try {
          const message = JSON.parse(line);
          const pending = controller.pending.get(message.id);
          if (pending) {
            if (typeof message.progress === "number" && !message.error && !message.result) {
              if (pending.onProgress) {
                pending.onProgress(message);
              }
            } else {
              controller.pending.delete(message.id);
              if (message.error) {
                pending.reject(new Error(message.error));
              } else {
                pending.resolve(message);
              }
            }
          }
        } catch (error) {
          resetPending(controller, new Error(`${controller.name} worker returned invalid JSON.`));
        }
      }

      newlineIndex = controller.buffer.indexOf("\n");
    }
  });

  controller.process.stderr.on("data", (chunk) => {
    const output = chunk.toString().trim();
    if (output) {
      console.error(output);
    }
  });

  controller.process.on("error", (error) => {
    resetPending(controller, error);
    controller.process = null;
  });

  controller.process.on("close", (code) => {
    resetPending(controller, new Error(`${controller.name} worker exited with code ${code}.`));
    controller.process = null;
  });

  return controller.process;
}

function requestWorker(controller, payload, options = {}) {
  return new Promise((resolve, reject) => {
    const child = startWorker(controller);
    const id = `${++controller.counter}-${randomUUID()}`;
    controller.pending.set(id, { resolve, reject, onProgress: options.onProgress });
    child.stdin.write(`${JSON.stringify({ id, ...payload })}\n`);
  });
}

function stopWorker(controller) {
  if (!controller.process) {
    return;
  }

  const child = controller.process;
  controller.process = null;
  controller.buffer = "";
  resetPending(controller, new Error(`${controller.name} worker was cleared.`));
  child.kill("SIGTERM");
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function runProcess(command, args, cwd = workRoot) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(new Error(stderr.trim() || `${command} exited with code ${code}.`));
    });
  });
}

function parseFps(value) {
  if (!value || typeof value !== "string") {
    return 0;
  }
  const [num, den] = value.split("/").map(Number);
  if (num && den) {
    return num / den;
  }
  return Number(value) || 0;
}

async function probeVideo(filePath) {
  const { stdout } = await runProcess("ffprobe", [
    "-v",
    "error",
    "-show_entries",
    "stream=width,height,r_frame_rate:format=duration",
    "-of",
    "json",
    filePath,
  ]);
  const payload = JSON.parse(stdout || "{}");
  const stream = payload.streams?.find((item) => item.width && item.height) || {};
  return {
    width: Number(stream.width) || 0,
    height: Number(stream.height) || 0,
    duration: Number(payload.format?.duration) || 0,
    fps: parseFps(stream.r_frame_rate),
  };
}

async function probeStill(filePath) {
  const { stdout } = await runProcess("ffprobe", [
    "-v",
    "error",
    "-show_entries",
    "stream=width,height",
    "-of",
    "json",
    filePath,
  ]);
  const payload = JSON.parse(stdout || "{}");
  const stream = payload.streams?.[0] || {};
  return {
    width: Number(stream.width) || 0,
    height: Number(stream.height) || 0,
  };
}

function toDataUrl(buffer, extension = ".jpg") {
  const mime = extension === ".png" ? "image/png" : "image/jpeg";
  return `data:${mime};base64,${buffer.toString("base64")}`;
}

function decodeDataUrl(dataUrl) {
  const index = dataUrl.indexOf(",");
  if (index < 0) {
    throw new Error("Expected a data URL.");
  }
  return Buffer.from(dataUrl.slice(index + 1), "base64");
}

function clampInteger(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.round(parsed)));
}

function clampFloat(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function normalizeEngine(value) {
  const normalized = String(value || "").toLowerCase();
  if (normalized === "rvm") {
    return "rvm";
  }
  if (normalized === "ppmattingv2") {
    return "ppmattingv2";
  }
  if (normalized === "modnet") {
    return "modnet";
  }
  if (normalized === "coreml_quality") {
    return "coreml_quality";
  }
  return "coreml_preview";
}

function getEngineFamily(engine) {
  if (engine === "rvm") {
    return "rvm";
  }
  if (engine === "ppmattingv2") {
    return "ppmattingv2";
  }
  if (engine === "modnet") {
    return "modnet";
  }
  return "coreml";
}

function getCoreMlProfile(engine) {
  return engine === "coreml_quality"
    ? { mode: "recurrent_matting", inputWidth: 384, inputHeight: 384, maxDimension: 1440, outputFormat: "jpg" }
    : { mode: "matting_lite", inputWidth: 128, inputHeight: 128, maxDimension: 960, outputFormat: "jpg" };
}

function getRequestedEngine(url, body = {}, headers = {}) {
  return normalizeEngine(url.searchParams.get("engine") || body.engine || headers["x-engine"]);
}

function createCoreMlSession(engine = "coreml_preview") {
  const sessionId = randomUUID();
  engineSessions.set(sessionId, { engine });
  return { session: sessionId };
}

async function createEngineSession(engine) {
  if (getEngineFamily(engine) === "rvm") {
    const response = await requestWorker(rvmWorker, { action: "create_session" });
    engineSessions.set(response.session, { engine: "rvm" });
    return response;
  }
  return createCoreMlSession(engine);
}

async function resetEngineSession(sessionId) {
  const entry = engineSessions.get(sessionId);
  if (!entry || getEngineFamily(entry.engine) !== "rvm") {
    engineSessions.set(sessionId, { engine: entry?.engine || "coreml_preview" });
    return { session: sessionId, reset: true };
  }

  return requestWorker(rvmWorker, {
    action: "reset_session",
    session: sessionId,
  });
}

async function closeEngineSession(sessionId) {
  const entry = engineSessions.get(sessionId);
  engineSessions.delete(sessionId);

  if (!entry || getEngineFamily(entry.engine) !== "rvm") {
    return { session: sessionId, closed: true };
  }

  return requestWorker(rvmWorker, {
    action: "close_session",
    session: sessionId,
  });
}

function stripDataUrlPrefix(value) {
  const index = value.indexOf(",");
  return index >= 0 ? value.slice(index + 1) : value;
}

async function matteFrameForSession(sessionId, imageBuffer, options = {}) {
  const entry = engineSessions.get(sessionId);
  const engine = entry?.engine || "coreml_preview";

  if (getEngineFamily(engine) === "rvm") {
    return requestWorker(rvmWorker, {
      action: "matte_frame",
      session: sessionId,
      image: imageBuffer.toString("base64"),
      downsample_ratio: clampFloat(options.downsampleRatio, 0.15, 1, 0.35),
    });
  }

  if (getEngineFamily(engine) === "modnet") {
    return requestWorker(modnetWorker, {
      action: "process",
      image: toDataUrl(imageBuffer, ".jpg"),
    }).then((result) => ({
      mask: stripDataUrlPrefix(result.alphaImage),
      cutout: stripDataUrlPrefix(result.cutoutImage),
      coverage: Number(result.summary?.alpha?.mean) || 0,
      max_alpha: Math.round((Number(result.summary?.alpha?.max) || 0) * 255),
      downsample_ratio: 1,
    }));
  }

  if (getEngineFamily(engine) === "ppmattingv2") {
    return requestWorker(ppmattingWorker, {
      action: "process",
      image: toDataUrl(imageBuffer, ".jpg"),
    }).then((result) => ({
      mask: stripDataUrlPrefix(result.alphaImage),
      cutout: stripDataUrlPrefix(result.cutoutImage),
      coverage: Number(result.summary?.alpha?.mean) || 0,
      max_alpha: Math.round((Number(result.summary?.alpha?.max) || 0) * 255),
      downsample_ratio: 1,
    }));
  }

  const profile = getCoreMlProfile(engine);
  const result = await requestWorker(coremlWorker, {
    action: "process",
    image: toDataUrl(imageBuffer, ".jpg"),
    mode: profile.mode,
    width: profile.inputWidth,
    height: profile.inputHeight,
  });

  return {
    mask: stripDataUrlPrefix(result.alphaImage),
    cutout: stripDataUrlPrefix(result.cutoutImage),
    coverage: Number(result.summary?.alpha?.mean) || 0,
    max_alpha: Math.round((Number(result.summary?.alpha?.max) || 0) * 255),
    downsample_ratio: 1,
  };
}

async function getHealth() {
  const [coreml, ppmattingv2, modnet, rvm] = await Promise.all([
    requestWorker(coremlWorker, { action: "health" }),
    requestWorker(ppmattingWorker, { action: "health" }),
    requestWorker(modnetWorker, { action: "health" }),
    requestWorker(rvmWorker, { action: "health" }),
  ]);

  return {
    ok: true,
    inputSize: coreml.inputSize,
    engines: {
      coreml,
      coreml_preview: coreml,
      coreml_quality: coreml,
      ppmattingv2,
      modnet,
      rvm,
    },
  };
}

async function processImage(body) {
  const engine = normalizeEngine(body.engine);

  if (getEngineFamily(engine) === "modnet") {
    return requestWorker(modnetWorker, {
      action: "process",
      image: body.image,
    });
  }

  if (getEngineFamily(engine) === "ppmattingv2") {
    return requestWorker(ppmattingWorker, {
      action: "process",
      image: body.image,
    });
  }

  if (getEngineFamily(engine) === "coreml") {
    const profile = getCoreMlProfile(engine);
    return requestWorker(coremlWorker, {
      action: "process",
      image: body.image,
      mode: profile.mode,
      width: profile.inputWidth,
      height: profile.inputHeight,
    });
  }

  const session = await requestWorker(rvmWorker, { action: "create_session" });
  try {
    const result = await requestWorker(rvmWorker, {
      action: "matte_frame",
      session: session.session,
      image: decodeDataUrl(body.image).toString("base64"),
      downsample_ratio: clampFloat(body.rvmDetail, 0.15, 1, 0.35),
    });
    return {
      alphaImage: `data:image/png;base64,${result.mask}`,
      cutoutImage: `data:image/png;base64,${result.cutout}`,
      compositeImage: `data:image/png;base64,${result.cutout}`,
      summary: {
        alpha: {
          min: 0,
          max: result.max_alpha / 255,
          mean: result.coverage,
        },
      },
      sourceSize: body.sourceSize || null,
      engine: "rvm",
    };
  } finally {
    await requestWorker(rvmWorker, {
      action: "close_session",
      session: session.session,
    }).catch(() => {});
  }
}

async function processVideo(inputPath, filename, options = {}, onProgress = null) {
  const engine = normalizeEngine(options.engine);
  const jobId = randomUUID();
  const jobDir = path.join(workRoot, jobId);
  const sourceFramesDir = path.join(jobDir, "source-frames");
  const outputFramesDir = path.join(jobDir, "output-frames");
  const outputPath = path.join(outputRoot, `${jobId}.mp4`);

  fs.mkdirSync(jobDir, { recursive: true });
  fs.mkdirSync(sourceFramesDir, { recursive: true });
  fs.mkdirSync(outputFramesDir, { recursive: true });

  try {
    const metadata = await probeVideo(inputPath);
    if (metadata.duration > maxVideoDurationSeconds) {
      throw new Error("Video is too long. The current maximum length is 2 hours.");
    }
    const requestedFps = clampInteger(options.processFps, 6, 60, metadata.fps || 12);
    const processFps = Math.max(1, requestedFps);
    const scaleFilter = getEngineFamily(engine) === "coreml"
      ? buildScaleFilter(getCoreMlProfile(engine).maxDimension)
      : buildScaleFilter(rvmBatchMaxDimension);

    await runProcess("ffmpeg", [
      "-y",
      "-i",
      inputPath,
      "-vf",
      `fps=${processFps},${scaleFilter}`,
      path.join(sourceFramesDir, "frame-%06d.jpg"),
    ]);

    if (onProgress) {
      onProgress({
        phase: "extracting_frames",
        progress: 12,
      });
    }

    const frameFiles = fs
      .readdirSync(sourceFramesDir)
      .filter((name) => name.endsWith(".jpg"))
      .sort();

    if (!frameFiles.length) {
      throw new Error("No frames were extracted from the video.");
    }

    let preview = null;
    let frameSize = { width: 0, height: 0 };
    let workerElapsedSeconds = 0;
    if (getEngineFamily(engine) === "coreml") {
      const profile = getCoreMlProfile(engine);
      const result = await requestWorker(coremlWorker, {
        action: "process_directory",
        inputDir: sourceFramesDir,
        outputDir: outputFramesDir,
        mode: profile.mode,
        width: profile.inputWidth,
        height: profile.inputHeight,
        outputFormat: profile.outputFormat,
      }, {
        onProgress: (message) => {
          if (onProgress) {
            onProgress({
              phase: message.phase || "processing_frames",
              progress: 12 + (Number(message.progress) || 0) * 0.76,
              processedFrames: message.processedFrames,
              frameCount: message.frameCount,
            });
          }
        }
      });

      preview = result.preview;
      workerElapsedSeconds = Number(result.elapsedSeconds) || 0;
    } else if (getEngineFamily(engine) === "modnet") {
      const result = await requestWorker(modnetWorker, {
        action: "process_directory",
        inputDir: sourceFramesDir,
        outputDir: outputFramesDir,
      }, {
        onProgress: (message) => {
          if (onProgress) {
            onProgress({
              phase: message.phase || "processing_frames",
              progress: 12 + (Number(message.progress) || 0) * 0.76,
              processedFrames: message.processedFrames,
              frameCount: message.frameCount,
            });
          }
        }
      });

      preview = result.preview;
      workerElapsedSeconds = Number(result.elapsedSeconds) || 0;
    } else if (getEngineFamily(engine) === "ppmattingv2") {
      const result = await requestWorker(ppmattingWorker, {
        action: "process_directory",
        inputDir: sourceFramesDir,
        outputDir: outputFramesDir,
      }, {
        onProgress: (message) => {
          if (onProgress) {
            onProgress({
              phase: message.phase || "processing_frames",
              progress: 12 + (Number(message.progress) || 0) * 0.76,
              processedFrames: message.processedFrames,
              frameCount: message.frameCount,
            });
          }
        }
      });

      preview = result.preview;
      workerElapsedSeconds = Number(result.elapsedSeconds) || 0;
    } else {
      const result = await requestWorker(rvmWorker, {
        action: "process_directory",
        inputDir: sourceFramesDir,
        outputDir: outputFramesDir,
        downsample_ratio: clampFloat(options.rvmDetail, 0.15, 1, 0.35),
      }, {
        onProgress: (message) => {
          if (onProgress) {
            onProgress({
              phase: message.phase || "processing_frames",
              progress: 12 + (Number(message.progress) || 0) * 0.76,
              processedFrames: message.processedFrames,
              frameCount: message.frameCount,
            });
          }
        }
      });

      preview = result.preview;
      workerElapsedSeconds = Number(result.elapsedSeconds) || 0;
    }

    const outputFrameExtension = getEngineFamily(engine) === "coreml"
      ? getCoreMlProfile(engine).outputFormat
      : "jpg";
    const outputFramePattern = `frame-%06d.${outputFrameExtension}`;
    const firstOutputFrame = path.join(outputFramesDir, `frame-000000.${outputFrameExtension}`);
    frameSize = await probeStill(firstOutputFrame);

    await runProcess("ffmpeg", [
      "-y",
      "-start_number",
      "0",
      "-framerate",
      String(processFps),
      "-i",
      path.join(outputFramesDir, outputFramePattern),
      "-vf",
      `scale=${metadata.width}:${metadata.height}`,
      "-r",
      String(processFps),
      "-c:v",
      "libx264",
      "-preset",
      "ultrafast",
      "-crf",
      "20",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      outputPath,
    ]);

    if (onProgress) {
      onProgress({
        phase: "encoding_preview",
        progress: 96,
      });
    }

    const outputMetadata = await probeVideo(outputPath);

    if (onProgress) {
      onProgress({
        phase: "completed",
        progress: 100,
      });
    }

    return {
      outputUrl: `/outputs/${jobId}.mp4`,
      filename: `${path.parse(filename || "preview").name}-${engine}-preview.mp4`,
      sourceMeta: metadata,
      outputMeta: outputMetadata,
      processFps,
      frameCount: frameFiles.length,
      workerElapsedSeconds,
      engine,
      preview,
    };
  } finally {
    fs.rmSync(jobDir, { recursive: true, force: true });
    const uploadDir = path.dirname(inputPath);
    if (path.basename(uploadDir).startsWith("upload-")) {
      fs.rmSync(uploadDir, { recursive: true, force: true });
    }
  }
}

function createVideoJob(inputPath, filename, options) {
  const jobId = randomUUID();
  const job = {
    id: jobId,
    status: "queued",
    progress: 0,
    phase: "queued",
    result: null,
    error: "",
    updatedAt: Date.now(),
  };
  videoJobs.set(jobId, job);

  processVideo(inputPath, filename, options, (update) => {
    job.status = "processing";
    job.progress = Math.max(job.progress, Math.min(100, Number(update.progress) || 0));
    job.phase = update.phase || job.phase;
    job.processedFrames = update.processedFrames || job.processedFrames || 0;
    job.frameCount = update.frameCount || job.frameCount || 0;
    job.updatedAt = Date.now();
  }).then((result) => {
    job.status = "completed";
    job.progress = 100;
    job.phase = "completed";
    job.result = result;
    job.updatedAt = Date.now();
  }).catch((error) => {
    job.status = "failed";
    job.error = error.message;
    job.updatedAt = Date.now();
  });

  return job;
}

async function clearCacheAndMemory() {
  engineSessions.clear();
  videoJobs.clear();
  exportJobs.clear();

  [coremlWorker, ppmattingWorker, modnetWorker, rvmWorker].forEach(stopWorker);
  await delay(150);
  cleanupWorkRootContents();
  fs.mkdirSync(outputRoot, { recursive: true });

  return { ok: true, cleared: true };
}

http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://127.0.0.1");

  if (req.method === "GET" && url.pathname === "/api/health") {
    try {
      const allHealth = await getHealth();
      const engine = getRequestedEngine(url, {}, req.headers);
      const selected = allHealth.engines[engine];
      sendJson(res, 200, engine === "rvm" ? selected : { ...selected, engines: allHealth.engines });
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/session") {
    try {
      const body = await readJsonBody(req);
      const engine = getRequestedEngine(url, body, req.headers);
      sendJson(res, 200, await createEngineSession(engine));
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "POST" && /^\/api\/session\/[^/]+\/reset$/.test(url.pathname)) {
    try {
      const sessionId = decodeURIComponent(url.pathname.split("/")[3]);
      sendJson(res, 200, await resetEngineSession(sessionId));
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "POST" && /^\/api\/session\/[^/]+\/matte$/.test(url.pathname)) {
    try {
      const sessionId = decodeURIComponent(url.pathname.split("/")[3]);
      const body = await readRequestBody(req, maxImageBytes);
      if (!body.length) {
        sendJson(res, 400, { error: "No image data received." });
        return;
      }
      const downsampleRatio = Number(url.searchParams.get("downsampleRatio"));
      const result = await matteFrameForSession(sessionId, body, { downsampleRatio });
      sendJson(res, 200, result);
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "DELETE" && /^\/api\/session\/[^/]+$/.test(url.pathname)) {
    try {
      const sessionId = decodeURIComponent(url.pathname.split("/")[3]);
      sendJson(res, 200, await closeEngineSession(sessionId));
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/export/start") {
    try {
      const body = await readJsonBody(req);
      const exportId = randomUUID();
      const frameDir = path.join(workRoot, exportId);
      fs.mkdirSync(frameDir, { recursive: true });
      exportJobs.set(exportId, {
        frameDir,
        purpose: body.purpose === "preview" ? "preview" : "export",
        frameCount: Number(body.frameCount) || 0,
        fps: Number(body.fps) || 24,
      });
      sendJson(res, 200, { exportId });
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "POST" && /^\/api\/export\/[^/]+\/frame$/.test(url.pathname)) {
    try {
      const exportId = decodeURIComponent(url.pathname.split("/")[3]);
      const job = exportJobs.get(exportId);
      if (!job) {
        sendJson(res, 404, { error: "Export job not found." });
        return;
      }
      const index = Number(url.searchParams.get("index"));
      if (!Number.isInteger(index) || index < 0) {
        sendJson(res, 400, { error: "A valid frame index is required." });
        return;
      }
      const body = await readRequestBody(req, maxImageBytes);
      const framePath = path.join(job.frameDir, `frame-${String(index).padStart(6, "0")}.png`);
      fs.writeFileSync(framePath, body);
      sendJson(res, 200, { ok: true });
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "POST" && /^\/api\/export\/[^/]+\/finalize$/.test(url.pathname)) {
    try {
      const exportId = decodeURIComponent(url.pathname.split("/")[3]);
      const job = exportJobs.get(exportId);
      if (!job) {
        sendJson(res, 404, { error: "Export job not found." });
        return;
      }
      const outputPath = path.join(outputRoot, `${exportId}.mp4`);
      await runProcess("ffmpeg", [
        "-y",
        "-start_number",
        "0",
        "-framerate",
        String(job.fps),
        "-i",
        path.join(job.frameDir, "frame-%06d.png"),
        "-r",
        String(job.fps),
        "-c:v",
        "libx264",
        "-preset",
        job.purpose === "preview" ? "ultrafast" : "slow",
        ...(job.purpose === "preview"
          ? ["-crf", "20", "-pix_fmt", "yuv420p"]
          : ["-crf", "0", "-pix_fmt", "yuv444p"]),
        "-movflags",
        "+faststart",
        outputPath,
      ]);
      fs.rmSync(job.frameDir, { recursive: true, force: true });
      exportJobs.delete(exportId);
      const metadata = await probeVideo(outputPath).catch(() => ({ duration: 0 }));
      sendJson(res, 200, {
        ok: true,
        filename: "backdrop-export.mp4",
        downloadUrl: `/exports/${exportId}.mp4`,
        duration: metadata.duration,
      });
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/process-image") {
    try {
      const body = await readJsonBody(req);
      if (!body.image) {
        sendJson(res, 400, { error: "Missing image payload." });
        return;
      }
      sendJson(res, 200, await processImage(body));
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/process-video") {
    try {
      const filenameHeader = String(req.headers["x-file-name"] || "upload.mp4");
      const safeName = path.basename(filenameHeader).replace(/[^\w.\-]+/g, "_");
      const engine = String(req.headers["x-engine"] || "coreml").toLowerCase();
      const processFps = req.headers["x-process-fps"];
      const rvmDetail = req.headers["x-rvm-detail"];
      const uploadId = randomUUID();
      const uploadDir = path.join(workRoot, `upload-${uploadId}`);
      const inputPath = path.join(uploadDir, safeName || "upload.mp4");
      fs.mkdirSync(uploadDir, { recursive: true });
      await streamRequestToFile(req, inputPath, maxUploadBytes);

      const result = await processVideo(inputPath, safeName, {
        engine,
        processFps,
        rvmDetail,
      });

      sendJson(res, 200, result);
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/process-video/start") {
    try {
      const filenameHeader = String(req.headers["x-file-name"] || "upload.mp4");
      const safeName = path.basename(filenameHeader).replace(/[^\w.\-]+/g, "_");
      const engine = String(req.headers["x-engine"] || "coreml").toLowerCase();
      const processFps = req.headers["x-process-fps"];
      const rvmDetail = req.headers["x-rvm-detail"];
      const uploadId = randomUUID();
      const uploadDir = path.join(workRoot, `upload-${uploadId}`);
      const inputPath = path.join(uploadDir, safeName || "upload.mp4");
      fs.mkdirSync(uploadDir, { recursive: true });
      await streamRequestToFile(req, inputPath, maxUploadBytes);

      const job = createVideoJob(inputPath, safeName, {
        engine,
        processFps,
        rvmDetail,
      });

      sendJson(res, 200, { jobId: job.id });
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/process-video/status") {
    const jobId = url.searchParams.get("jobId");
    if (!jobId || !videoJobs.has(jobId)) {
      sendJson(res, 404, { error: "Video job not found." });
      return;
    }

    const job = videoJobs.get(jobId);
    sendJson(res, 200, {
      jobId: job.id,
      status: job.status,
      progress: job.progress,
      phase: job.phase,
      processedFrames: job.processedFrames || 0,
      frameCount: job.frameCount || 0,
      result: job.result,
      error: job.error,
    });
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/clear-cache") {
    try {
      sendJson(res, 200, await clearCacheAndMemory());
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return;
  }

  if (req.method === "GET" && url.pathname.startsWith("/outputs/")) {
    const filePath = path.normalize(path.join(outputRoot, url.pathname.replace("/outputs/", "")));
    if (!filePath.startsWith(outputRoot)) {
      send(res, 403, "Forbidden");
      return;
    }
    sendFile(filePath, res);
    return;
  }

  if (req.method === "GET" && url.pathname.startsWith("/exports/")) {
    const filePath = path.normalize(path.join(outputRoot, url.pathname.replace("/exports/", "")));
    if (!filePath.startsWith(outputRoot)) {
      send(res, 403, "Forbidden");
      return;
    }
    sendFile(filePath, res);
    return;
  }

  const requestPath = url.pathname === "/" ? "/index.html" : url.pathname;
  const filePath = path.normalize(path.join(root, requestPath));
  if (!filePath.startsWith(root)) {
    send(res, 403, "Forbidden");
    return;
  }

  sendFile(filePath, res);
}).listen(port, host, () => {
  console.log(`Backdrop server running at http://${host}:${port}`);
});
