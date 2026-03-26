import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
MODNET_ROOT = ROOT / "vendor" / "MODNet"
sys.path.insert(0, str(MODNET_ROOT))

from src.models.modnet import MODNet  # noqa: E402


MODEL_PATH = MODNET_ROOT / "pretrained" / "modnet_webcam_portrait_matting.ckpt"
BACKGROUND_RGBA = (244, 239, 230, 255)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TORCH_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def select_device():
    force_cpu = os.environ.get("BGREMOVER_FORCE_CPU") == "1"
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_image(image, fmt="PNG", **save_kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def matte_summary(alpha_image):
    alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
    return {
        "min": float(alpha.min()) if alpha.size else 0.0,
        "max": float(alpha.max()) if alpha.size else 0.0,
        "mean": float(alpha.mean()) if alpha.size else 0.0,
    }


def resize_target(width, height, max_side=512):
    if width >= height:
        resized_height = max_side
        resized_width = int(width / height * max_side)
    else:
        resized_width = max_side
        resized_height = int(height / width * max_side)
    resized_width -= resized_width % 32
    resized_height -= resized_height % 32
    return max(32, resized_width), max(32, resized_height)


class ModNetService:
    def __init__(self):
        self.device = select_device()
        self.model = self._load_model()

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing MODNet checkpoint at {MODEL_PATH}.")

        model = MODNet(backbone_pretrained=False)
        weights = torch.load(MODEL_PATH, map_location="cpu")
        if any(key.startswith("module.") for key in weights.keys()):
            weights = {key.replace("module.", "", 1): value for key, value in weights.items()}
        model.load_state_dict(weights)
        model = model.to(self.device).eval()
        return model

    def health(self):
        return {
            "ok": True,
            "provider": "modnet",
            "device": str(self.device),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        }

    def matte_image(self, image_bytes, max_side=512):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        resized_width, resized_height = resize_target(width, height, max_side=max_side)

        resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        image_tensor = TORCH_TRANSFORMS(resized_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            _, _, matte_tensor = self.model(image_tensor, True)

        alpha = matte_tensor[0, 0].detach().to("cpu").clamp(0, 1).numpy()
        alpha_image = Image.fromarray((alpha * 255).astype(np.uint8), mode="L").resize(
            (width, height), Image.Resampling.LANCZOS
        )

        cutout = image.copy()
        cutout.putalpha(alpha_image)

        return {
            "alphaImage": encode_image(alpha_image, "PNG"),
            "cutoutImage": encode_image(cutout, "PNG"),
            "compositeImage": encode_image(
                Image.alpha_composite(Image.new("RGBA", image.size, BACKGROUND_RGBA), cutout).convert("RGB"),
                "PNG",
            ),
            "summary": {
                "alpha": matte_summary(alpha_image),
            },
            "sourceSize": {"width": width, "height": height},
            "engine": "modnet",
        }

    def process_directory(self, input_dir, output_dir, max_side=512):
        in_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = sorted(path for path in in_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not frame_paths:
            raise ValueError("No source frames found in the input directory.")

        started_at = time.perf_counter()
        preview = None

        for index, frame_path in enumerate(frame_paths):
            result = self.matte_image(frame_path.read_bytes(), max_side=max_side)
            cutout_image = Image.open(io.BytesIO(base64.b64decode(result["cutoutImage"].split(",", 1)[1]))).convert("RGBA")
            composited = Image.alpha_composite(
                Image.new("RGBA", cutout_image.size, BACKGROUND_RGBA),
                cutout_image,
            ).convert("RGB")
            composited.save(
                out_dir / f"frame-{index:06d}.jpg",
                format="JPEG",
                quality=95,
                subsampling=0,
                optimize=True,
            )

            if preview is None:
                preview = result

            emit(
                {
                    "id": self.request_id,
                    "progress": round(((index + 1) / len(frame_paths)) * 100, 2),
                    "processedFrames": index + 1,
                    "frameCount": len(frame_paths),
                    "phase": "processing_frames",
                }
            )

        return {
            "frameCount": len(frame_paths),
            "elapsedSeconds": time.perf_counter() - started_at,
            "preview": preview,
        }


SERVICE = ModNetService()


def handle_message(line):
    payload = json.loads(line)
    action = payload["action"]
    request_id = payload["id"]
    SERVICE.request_id = request_id

    if action == "health":
        return {"id": request_id, **SERVICE.health()}

    if action == "process":
        image_value = payload["image"]
        if "," in image_value:
            image_value = image_value.split(",", 1)[1]
        image_bytes = base64.b64decode(image_value)
        return {"id": request_id, **SERVICE.matte_image(image_bytes)}

    if action == "process_directory":
        return {
            "id": request_id,
            **SERVICE.process_directory(payload["inputDir"], payload["outputDir"]),
        }

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
