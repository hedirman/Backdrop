import base64
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
RVM_ROOT = ROOT / "vendor" / "RobustVideoMatting"
sys.path.insert(0, str(RVM_ROOT))

from model import MattingNetwork  # noqa: E402


MODEL_PATH = ROOT / "models" / "rvm_mobilenetv3.pth"


def select_device():
    force_cpu = os.environ.get("BGREMOVER_FORCE_CPU") == "1"
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_downsample_ratio(height, width):
    return min(512 / max(height, width), 1)


def encode_image(image, fmt="PNG", **save_kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


class RvmService:
    def __init__(self):
        self.device = select_device()
        self.variant = "mobilenetv3"
        self.model = self._load_model()
        self.sessions = {}

    def _load_model(self):
        model = MattingNetwork(self.variant, pretrained_backbone=False).eval()
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        return model

    def health(self):
        return {
            "ok": True,
            "provider": "rvm",
            "variant": self.variant,
            "device": str(self.device),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        }

    def create_session(self):
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = [None, None, None, None]
        return {"session": session_id}

    def reset_session(self, session_id):
        self.sessions[session_id] = [None, None, None, None]
        return {"session": session_id, "reset": True}

    def close_session(self, session_id):
        self.sessions.pop(session_id, None)
        return {"session": session_id, "closed": True}

    def matte_frame(self, session_id, image_bytes, downsample_ratio=None):
        if session_id not in self.sessions:
            self.sessions[session_id] = [None, None, None, None]

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        ratio = downsample_ratio if downsample_ratio is not None else default_downsample_ratio(height, width)
        ratio = float(max(0.1, min(1.0, ratio)))

        src = TF.to_tensor(image).unsqueeze(0).unsqueeze(0).to(self.device)
        rec = self.sessions[session_id]

        with torch.inference_mode():
            fgr, pha, *new_rec = self.model(src, *rec, ratio)

        self.sessions[session_id] = [state.detach() if state is not None else None for state in new_rec]

        alpha = pha[0, 0, 0].detach().to("cpu").numpy()
        alpha = np.clip(alpha, 0, 1)
        alpha_u8 = (alpha * 255).astype(np.uint8)
        foreground = fgr[0, 0].detach().to("cpu").permute(1, 2, 0).numpy()
        foreground = np.clip(foreground, 0, 1)
        foreground_u8 = (foreground * 255).astype(np.uint8)
        cutout_rgba = np.dstack([foreground_u8, alpha_u8])

        mask_buffer = io.BytesIO()
        cutout_buffer = io.BytesIO()
        Image.fromarray(alpha_u8).save(mask_buffer, format="PNG")
        Image.fromarray(cutout_rgba).save(cutout_buffer, format="PNG")

        return {
            "mask": base64.b64encode(mask_buffer.getvalue()).decode("ascii"),
            "cutout": base64.b64encode(cutout_buffer.getvalue()).decode("ascii"),
            "coverage": float(alpha.mean()),
            "max_alpha": int(alpha_u8.max()),
            "downsample_ratio": ratio,
        }

    def process_directory(self, input_dir, output_dir, downsample_ratio=None):
        in_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = sorted(path for path in in_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not frame_paths:
            raise ValueError("No source frames found in the input directory.")

        session = self.create_session()["session"]
        started_at = time.perf_counter()
        preview = None

        try:
            for index, frame_path in enumerate(frame_paths):
                result = self.matte_frame(session, frame_path.read_bytes(), downsample_ratio)
                cutout_image = Image.open(io.BytesIO(base64.b64decode(result["cutout"]))).convert("RGBA")
                background = Image.new("RGBA", cutout_image.size, (244, 239, 230, 255))
                composited = Image.alpha_composite(background, cutout_image).convert("RGB")
                composited.save(
                    out_dir / f"frame-{index:06d}.jpg",
                    format="JPEG",
                    quality=95,
                    subsampling=0,
                    optimize=True,
                )

                if preview is None:
                    preview = {
                        "alphaImage": encode_image(Image.open(io.BytesIO(base64.b64decode(result["mask"]))).convert("L"), "PNG"),
                        "cutoutImage": encode_image(cutout_image, "PNG"),
                        "compositeImage": encode_image(composited, "PNG"),
                        "summary": {
                            "alpha": {
                                "min": 0,
                                "max": result["max_alpha"] / 255,
                                "mean": result["coverage"],
                            },
                        },
                        "engine": "rvm",
                    }

            elapsed_seconds = time.perf_counter() - started_at
            return {
                "frameCount": len(frame_paths),
                "elapsedSeconds": elapsed_seconds,
                "preview": preview,
            }
        finally:
            self.close_session(session)


SERVICE = RvmService()


def handle_message(line):
    payload = json.loads(line)
    action = payload["action"]
    request_id = payload["id"]

    if action == "health":
        return {"id": request_id, **SERVICE.health()}

    if action == "create_session":
        return {"id": request_id, **SERVICE.create_session()}

    if action == "reset_session":
        return {"id": request_id, **SERVICE.reset_session(payload["session"])}

    if action == "close_session":
        return {"id": request_id, **SERVICE.close_session(payload["session"])}

    if action == "matte_frame":
        result = SERVICE.matte_frame(
            payload["session"],
            base64.b64decode(payload["image"]),
            payload.get("downsample_ratio"),
        )
        return {"id": request_id, **result}

    if action == "process_directory":
        in_dir = Path(payload["inputDir"])
        out_dir = Path(payload["outputDir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = sorted(path for path in in_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not frame_paths:
            raise ValueError("No source frames found in the input directory.")

        session = SERVICE.create_session()["session"]
        started_at = time.perf_counter()
        preview = None

        try:
            for index, frame_path in enumerate(frame_paths):
                result = SERVICE.matte_frame(session, frame_path.read_bytes(), payload.get("downsample_ratio"))
                cutout_image = Image.open(io.BytesIO(base64.b64decode(result["cutout"]))).convert("RGBA")
                background = Image.new("RGBA", cutout_image.size, (244, 239, 230, 255))
                composited = Image.alpha_composite(background, cutout_image).convert("RGB")
                composited.save(
                    out_dir / f"frame-{index:06d}.jpg",
                    format="JPEG",
                    quality=95,
                    subsampling=0,
                    optimize=True,
                )

                if preview is None:
                    preview = {
                        "alphaImage": encode_image(Image.open(io.BytesIO(base64.b64decode(result["mask"]))).convert("L"), "PNG"),
                        "cutoutImage": encode_image(cutout_image, "PNG"),
                        "compositeImage": encode_image(composited, "PNG"),
                        "summary": {
                            "alpha": {
                                "min": 0,
                                "max": result["max_alpha"] / 255,
                                "mean": result["coverage"],
                            },
                        },
                        "engine": "rvm",
                    }

                emit({
                    "id": request_id,
                    "progress": round(((index + 1) / len(frame_paths)) * 100, 2),
                    "processedFrames": index + 1,
                    "frameCount": len(frame_paths),
                    "phase": "processing_frames",
                })

            return {
                "id": request_id,
                "frameCount": len(frame_paths),
                "elapsedSeconds": time.perf_counter() - started_at,
                "preview": preview,
            }
        finally:
            SERVICE.close_session(session)

    raise ValueError(f"Unsupported action: {action}")


def emit(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request_id = "unknown"
        try:
            request_id = json.loads(line).get("id", request_id)
            emit(handle_message(line))
        except Exception as error:
            emit({"id": request_id, "error": str(error)})


if __name__ == "__main__":
    main()
