import argparse
import base64
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image
from PIL import ImageFilter


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
MODEL_TEMPLATE = "rvm_single_frame_matting_lite_{height}x{width}.mlpackage"
TMPDIR = HERE / "tmp"


def parse_args():
    parser = argparse.ArgumentParser(description="Sandbox Core ML worker")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    return parser.parse_args()


def model_path_for(mode: str, width: int, height: int):
    if mode == "recurrent_matting":
        template = "rvm_recurrent_frame_matting_{height}x{width}.mlpackage"
    elif mode == "matting":
        template = "rvm_single_frame_matting_{height}x{width}.mlpackage"
    else:
        template = MODEL_TEMPLATE
    path = ARTIFACTS / template.format(height=height, width=width)
    if not path.exists():
        raise FileNotFoundError(f"Core ML package not found for {mode} {width}x{height}: {path}")
    return path


def load_model(model_path: str):
    TMPDIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(TMPDIR)
    tempfile.tempdir = str(TMPDIR)
    return ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)


def decode_image(payload: str):
    if "," in payload:
        _, encoded = payload.split(",", 1)
    else:
        encoded = payload
    raw = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return image


def encode_image(image: Image.Image, fmt: str = "PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def metadata_path_for(mode: str, width: int, height: int):
    if mode == "recurrent_matting":
        filename = f"rvm_recurrent_frame_matting_{height}x{width}.json"
    elif mode == "matting":
        filename = f"rvm_single_frame_matting_{height}x{width}.json"
    else:
        filename = f"rvm_single_frame_matting_lite_{height}x{width}.json"
    return ARTIFACTS / filename


def recurrent_state_shapes_for(width: int, height: int):
    metadata_path = metadata_path_for("recurrent_matting", width, height)
    metadata = json.loads(metadata_path.read_text())
    return [tuple(shape) for shape in metadata["state_shapes"]]


def to_input_tensor(image: Image.Image, width: int, height: int):
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)[None, None, :, :, :].astype(np.float32), resized


def to_recurrent_input_tensor(image: Image.Image, width: int, height: int):
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)[None, :, :, :].astype(np.float32), resized


def upscale_map(single_channel: np.ndarray, target_size):
    image = Image.fromarray(np.clip(single_channel * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    return image.resize(target_size, Image.Resampling.BILINEAR)


def upscale_rgb(rgb: np.ndarray, target_size):
    image = Image.fromarray(np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    return image.resize(target_size, Image.Resampling.BILINEAR)


def refine_alpha_image(alpha_image: Image.Image, previous_alpha_image: Image.Image | None = None):
    # Opening removes tiny bright speckles, closing fills pinholes.
    cleaned = alpha_image.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    cleaned = cleaned.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    cleaned = cleaned.filter(ImageFilter.GaussianBlur(radius=0.8))

    alpha = np.asarray(cleaned, dtype=np.float32)
    alpha[alpha < 10] = 0
    alpha[alpha > 245] = 255

    if previous_alpha_image is not None:
      previous = np.asarray(previous_alpha_image, dtype=np.float32)
      uncertain = (alpha > 8) & (alpha < 247)
      alpha[uncertain] = previous[uncertain] * 0.22 + alpha[uncertain] * 0.78

    return Image.fromarray(np.clip(alpha, 0.0, 255.0).astype(np.uint8), mode="L")


def compose_quality_foreground(source: Image.Image, model_foreground: Image.Image, alpha_image: Image.Image):
    source_rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    foreground_rgb = np.asarray(model_foreground.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
    confidence = np.clip((alpha - 0.18) / 0.62, 0.0, 1.0)[..., None]
    # At uncertain boundaries, trust the original source color more.
    blended = source_rgb * (1.0 - confidence) + foreground_rgb * confidence
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), mode="RGB")


def composite_cutout(source: Image.Image, alpha_image: Image.Image):
    rgba = source.convert("RGBA")
    rgba.putalpha(alpha_image)
    return rgba


def composite_on_background(source: Image.Image, alpha_image: Image.Image, color=(238, 234, 224)):
    foreground = source.convert("RGBA")
    foreground.putalpha(alpha_image)
    background = Image.new("RGBA", foreground.size, color + (255,))
    return Image.alpha_composite(background, foreground).convert("RGB")


def summarize(alpha: np.ndarray, foreground_residual: np.ndarray):
    return {
        "alpha": {
            "min": float(alpha.min()),
            "max": float(alpha.max()),
            "mean": float(alpha.mean()),
        },
        "foreground_residual": {
            "min": float(foreground_residual.min()),
            "max": float(foreground_residual.max()),
            "mean": float(foreground_residual.mean()),
        },
    }


def predict_image_object(
    model,
    source: Image.Image,
    width: int,
    height: int,
    mode: str = "matting_lite",
    state=None,
    previous_alpha_image: Image.Image | None = None,
):
    if mode == "recurrent_matting":
        tensor, resized = to_recurrent_input_tensor(source, width, height)
        if state is None:
            state = [np.zeros(shape, dtype=np.float32) for shape in recurrent_state_shapes_for(width, height)]
        outputs = model.predict({
            "src": tensor,
            "r1": state[0],
            "r2": state[1],
            "r3": state[2],
            "r4": state[3],
        })
    else:
        tensor, resized = to_input_tensor(source, width, height)
        outputs = model.predict({"src": tensor})

    if mode == "recurrent_matting":
        alpha = np.asarray(outputs["alpha"])[0, 0]
    else:
        alpha = np.asarray(outputs["alpha"])[0, 0]
    foreground_residual = None
    foreground = None
    if "foreground_residual" in outputs:
        foreground_residual = np.asarray(outputs["foreground_residual"])[0].transpose(1, 2, 0)
    if "foreground" in outputs:
        foreground = np.asarray(outputs["foreground"])[0].transpose(1, 2, 0)

    alpha_up = upscale_map(alpha, source.size)
    if mode in {"matting", "recurrent_matting"}:
        alpha_up = refine_alpha_image(alpha_up, previous_alpha_image=previous_alpha_image)
    if foreground is not None:
        foreground_up = upscale_rgb(np.clip(foreground, 0.0, 1.0), source.size)
        quality_foreground = compose_quality_foreground(source, foreground_up, alpha_up)
        cutout = composite_cutout(quality_foreground, alpha_up)
        composite = composite_on_background(quality_foreground, alpha_up)
        foreground_summary = foreground
    else:
        foreground_up = upscale_rgb(foreground_residual, source.size)
        # matting_lite gives us a residual-like foreground signal, which is useful
        # for diagnostics but not visually correct as a final subject preview.
        # For the sandbox preview, use the real source image with the predicted alpha.
        cutout = composite_cutout(source, alpha_up)
        composite = composite_on_background(source, alpha_up)
        foreground_summary = foreground_residual

    result = {
        "alphaImageObject": alpha_up,
        "foregroundResidualImageObject": foreground_up,
        "cutoutImageObject": cutout,
        "compositeImageObject": composite,
        "resizedSourceObject": resized,
        "summary": summarize(alpha, foreground_summary),
        "sourceSize": {"width": source.width, "height": source.height},
        "modelInputSize": {"width": width, "height": height},
    }
    if mode == "recurrent_matting":
        result["nextState"] = [
            np.asarray(outputs["nr1"]).astype(np.float32),
            np.asarray(outputs["nr2"]).astype(np.float32),
            np.asarray(outputs["nr3"]).astype(np.float32),
            np.asarray(outputs["nr4"]).astype(np.float32),
        ]
    return result


def process_image_object(model, source: Image.Image, width: int, height: int, mode: str = "matting_lite", state=None):
    result = predict_image_object(model, source, width, height, mode=mode, state=state)

    return {
        "alphaImage": encode_image(result["alphaImageObject"], "PNG"),
        "foregroundResidualImage": encode_image(result["foregroundResidualImageObject"], "PNG"),
        "cutoutImage": encode_image(result["cutoutImageObject"], "PNG"),
        "compositeImage": encode_image(result["compositeImageObject"], "PNG"),
        "resizedSource": encode_image(result["resizedSourceObject"], "PNG"),
        "summary": result["summary"],
        "sourceSize": result["sourceSize"],
        "modelInputSize": result["modelInputSize"],
    }


def process_image(model, data_url: str, width: int, height: int, mode: str = "matting_lite", state=None):
    source = decode_image(data_url)
    return process_image_object(model, source, width, height, mode=mode, state=state)


def process_directory(
    model,
    input_dir: str,
    output_dir: str,
    width: int,
    height: int,
    mode: str = "matting_lite",
    request_id=None,
    output_format: str = "jpg",
):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(path for path in in_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frame_paths:
      raise ValueError("No source frames found in the input directory.")

    started_at = time.perf_counter()
    preview = None
    state = None
    previous_alpha_image = None

    for index, frame_path in enumerate(frame_paths):
        source = Image.open(frame_path).convert("RGB")
        result = predict_image_object(
            model,
            source,
            width,
            height,
            mode=mode,
            state=state,
            previous_alpha_image=previous_alpha_image,
        )
        if mode == "recurrent_matting":
            state = result.get("nextState")
        previous_alpha_image = result["alphaImageObject"]
        frame_path = out_dir / f"frame-{index:06d}.{output_format}"
        if output_format == "png":
            result["compositeImageObject"].save(frame_path, format="PNG")
        else:
            result["compositeImageObject"].save(
                frame_path,
                format="JPEG",
                quality=95,
                subsampling=0,
                optimize=True,
            )

        if preview is None:
            preview = {
                "alphaImage": encode_image(result["alphaImageObject"], "PNG"),
                "foregroundResidualImage": encode_image(result["foregroundResidualImageObject"], "PNG"),
                "cutoutImage": encode_image(result["cutoutImageObject"], "PNG"),
                "compositeImage": encode_image(result["compositeImageObject"], "PNG"),
                "summary": result["summary"],
                "sourceSize": result["sourceSize"],
                "modelInputSize": result["modelInputSize"],
            }

        if request_id is not None:
            sys.stdout.write(json.dumps({
                "id": request_id,
                "progress": round(((index + 1) / len(frame_paths)) * 100, 2),
                "processedFrames": index + 1,
                "frameCount": len(frame_paths),
                "phase": "processing_frames",
            }) + "\n")
            sys.stdout.flush()

    elapsed_seconds = time.perf_counter() - started_at
    return {
        "frameCount": len(frame_paths),
        "elapsedSeconds": elapsed_seconds,
        "preview": preview,
    }


def main():
    args = parse_args()
    model_cache = {}

    def get_model(mode: str, width: int, height: int):
        key = (mode, width, height)
        if key not in model_cache:
            model_cache[key] = load_model(str(model_path_for(mode, width, height)))
        return model_cache[key]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
            request_id = payload.get("id")
            action = payload.get("action")

            if action == "health":
                response = {
                    "id": request_id,
                    "status": "ok",
                    "modelPath": str(model_path_for("matting_lite", args.width, args.height)),
                    "inputSize": {"width": args.width, "height": args.height},
                }
            elif action == "process":
                width = int(payload.get("width", args.width))
                height = int(payload.get("height", args.height))
                mode = str(payload.get("mode", "matting_lite"))
                response = {
                    "id": request_id,
                    **process_image(get_model(mode, width, height), payload["image"], width, height, mode=mode),
                }
            elif action == "process_directory":
                width = int(payload.get("width", args.width))
                height = int(payload.get("height", args.height))
                mode = str(payload.get("mode", "matting_lite"))
                response = {
                    "id": request_id,
                    **process_directory(
                        get_model(mode, width, height),
                        payload["inputDir"],
                        payload["outputDir"],
                        width,
                        height,
                        mode=mode,
                        request_id=request_id,
                        output_format=str(payload.get("outputFormat", "jpg")).lower(),
                    ),
                }
            else:
                raise ValueError(f"Unsupported action: {action}")
        except Exception as error:  # noqa: BLE001
            response = {
                "id": payload.get("id") if "payload" in locals() and isinstance(payload, dict) else None,
                "error": str(error),
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
