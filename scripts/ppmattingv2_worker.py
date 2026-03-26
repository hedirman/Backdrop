import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from paddle.inference import Config as PredictConfig
from paddle.inference import create_predictor

ROOT = Path(__file__).resolve().parents[1]
PADDLESEG_ROOT = ROOT / "vendor" / "PaddleSeg"
MATTING_ROOT = PADDLESEG_ROOT / "Matting"
MODEL_ROOT = ROOT / "models" / "ppmattingv2" / "ppmattingv2-stdc1-human_512"
SEG_HOME = ROOT / "tmp" / "paddleseg"

SEG_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SEG_HOME", str(SEG_HOME))

sys.path.insert(0, str(PADDLESEG_ROOT))
sys.path.insert(0, str(MATTING_ROOT))
sys.path.insert(0, str(MATTING_ROOT / "deploy" / "python"))

from paddleseg.cvlibs import manager  # noqa: E402

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting.transforms as T  # noqa: E402


BACKGROUND_RGBA = (244, 239, 230, 255)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PPMATTING_MAX_SHORT = 576
ALPHA_THRESHOLD = 34


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


class DeployConfig:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)
        self._dir = os.path.dirname(path)
        self._transforms = self.load_transforms(self.dic["Deploy"]["transforms"])

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic["Deploy"]["model"])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic["Deploy"]["params"])

    @staticmethod
    def load_transforms(transform_list):
        transforms = []
        for transform in transform_list:
            options = dict(transform)
            transform_type = options.pop("type")
            if transform_type == "LimitShort":
                options["max_short"] = max(int(options.get("max_short", 512)), PPMATTING_MAX_SHORT)
            transforms.append(manager.TRANSFORMS[transform_type](**options))
        return T.Compose(transforms)


class PPMattingV2Service:
    def __init__(self):
        deploy_yaml = MODEL_ROOT / "deploy.yaml"
        if not deploy_yaml.exists():
            raise FileNotFoundError(f"Missing PP-MattingV2 deploy config at {deploy_yaml}.")

        self.cfg = DeployConfig(str(deploy_yaml))
        self.predictor = self._build_predictor()
        self.input_names = self.predictor.get_input_names()
        self.input_handle = self.predictor.get_input_handle(self.input_names[0])
        self.output_names = self.predictor.get_output_names()
        self.output_handle = self.predictor.get_output_handle(self.output_names[0])
        self.request_id = "unknown"

    def _build_predictor(self):
        pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        pred_cfg.disable_gpu()
        pred_cfg.disable_glog_info()
        pred_cfg.enable_memory_optim()
        pred_cfg.switch_ir_optim(True)
        return create_predictor(pred_cfg)

    def health(self):
        return {
            "ok": True,
            "provider": "ppmattingv2",
            "device": "cpu",
            "model": f"PP-MattingV2-{PPMATTING_MAX_SHORT}",
        }

    def preprocess(self, image_rgb):
        data = {"img": image_rgb, "trans_info": []}
        data = self.cfg.transforms(data)
        return data

    def postprocess(self, alpha, trans_info):
        for item in trans_info[::-1]:
            if item[0] == "resize":
                height, width = item[1][0], item[1][1]
                alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
            elif item[0] == "padding":
                height, width = item[1][0], item[1][1]
                alpha = alpha[0:height, 0:width]
            else:
                raise ValueError(f"Unexpected transform info '{item[0]}'.")
        return np.clip(alpha, 0.0, 1.0)

    def _fill_small_holes(self, binary_mask):
        inverse = (~binary_mask).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
        if count <= 1:
            return binary_mask

        height, width = binary_mask.shape
        max_hole_area = max(24, int(height * width * 0.00045))
        filled = binary_mask.copy()

        for component_id in range(1, count):
            left = stats[component_id, cv2.CC_STAT_LEFT]
            top = stats[component_id, cv2.CC_STAT_TOP]
            comp_width = stats[component_id, cv2.CC_STAT_WIDTH]
            comp_height = stats[component_id, cv2.CC_STAT_HEIGHT]
            area = stats[component_id, cv2.CC_STAT_AREA]
            touches_edge = (
                left == 0
                or top == 0
                or left + comp_width >= width
                or top + comp_height >= height
            )
            if not touches_edge and area <= max_hole_area:
                filled[labels == component_id] = True

        return filled

    def _repair_temporal_dropouts(self, alpha_u8, previous_alpha_u8):
        if previous_alpha_u8 is None:
            return alpha_u8

        prev_strong = previous_alpha_u8 >= 190
        if not np.any(prev_strong):
            return alpha_u8

        dilated_prev = cv2.dilate(
            prev_strong.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)

        dropout = dilated_prev & (alpha_u8 <= 24)
        if not np.any(dropout):
            return alpha_u8

        count, labels, stats, _ = cv2.connectedComponentsWithStats(dropout.astype(np.uint8), connectivity=8)
        if count <= 1:
            return alpha_u8

        height, width = alpha_u8.shape
        max_patch_area = max(42, int(height * width * 0.0012))
        repaired = alpha_u8.copy()

        for component_id in range(1, count):
            area = stats[component_id, cv2.CC_STAT_AREA]
            if area > max_patch_area:
                continue
            mask = labels == component_id
            recovered = np.clip(previous_alpha_u8[mask].astype(np.float32) * 0.52, 0, 255).astype(np.uint8)
            repaired[mask] = np.maximum(repaired[mask], recovered)

        return repaired

    def _remove_isolated_foreground(self, alpha_u8, previous_alpha_u8):
        binary = alpha_u8 >= 26
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        if count <= 1:
            return alpha_u8

        height, width = alpha_u8.shape
        min_keep_area = max(20, int(height * width * 0.00018))
        previous_region = None
        if previous_alpha_u8 is not None:
            previous_region = cv2.dilate(
                (previous_alpha_u8 >= 110).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            ).astype(bool)

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component = int(np.argmax(areas)) + 1 if areas.size else None
        keep_mask = np.zeros_like(binary, dtype=bool)
        main_left = stats[largest_component, cv2.CC_STAT_LEFT] if largest_component else 0
        main_top = stats[largest_component, cv2.CC_STAT_TOP] if largest_component else 0
        main_width = stats[largest_component, cv2.CC_STAT_WIDTH] if largest_component else width
        main_height = stats[largest_component, cv2.CC_STAT_HEIGHT] if largest_component else height
        main_right = main_left + main_width
        main_bottom = main_top + main_height
        main_center_x = centroids[largest_component][0] if largest_component else width / 2
        horizontal_allowance = max(24, int(main_width * 0.45))
        vertical_allowance = max(28, int(main_height * 0.25))

        for component_id in range(1, count):
            area = stats[component_id, cv2.CC_STAT_AREA]
            if component_id == largest_component:
                keep_mask |= labels == component_id
                continue

            if area >= min_keep_area:
                component_mask = labels == component_id
                if previous_region is not None and np.any(previous_region & component_mask):
                    keep_mask |= component_mask
                    continue

                left = stats[component_id, cv2.CC_STAT_LEFT]
                top = stats[component_id, cv2.CC_STAT_TOP]
                comp_width = stats[component_id, cv2.CC_STAT_WIDTH]
                comp_height = stats[component_id, cv2.CC_STAT_HEIGHT]
                right = left + comp_width
                bottom = top + comp_height
                center_x = centroids[component_id][0]

                near_main_x = abs(center_x - main_center_x) <= (main_width / 2 + horizontal_allowance)
                near_main_y = top <= main_bottom + vertical_allowance and bottom >= main_top - vertical_allowance
                overlaps_expanded_body = (
                    right >= main_left - horizontal_allowance
                    and left <= main_right + horizontal_allowance
                    and bottom >= main_top - vertical_allowance
                    and top <= main_bottom + vertical_allowance
                )
                not_high_artifact = top >= max(8, int(main_top - main_height * 0.18))

                if (near_main_x and near_main_y or overlaps_expanded_body) and not_high_artifact:
                    keep_mask |= component_mask

        alpha_u8[~keep_mask] = 0
        return alpha_u8

    def refine_alpha(self, alpha, previous_alpha=None):
        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        previous_alpha_u8 = None
        if previous_alpha is not None:
            previous_alpha_u8 = np.clip(previous_alpha * 255.0, 0, 255).astype(np.uint8)

        binary = alpha_u8 >= ALPHA_THRESHOLD
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel_small, iterations=1).astype(bool)
        filled_binary = self._fill_small_holes(binary)
        fill_mask = filled_binary & ~binary
        alpha_u8[fill_mask] = np.maximum(alpha_u8[fill_mask], 160)

        alpha_u8 = self._repair_temporal_dropouts(alpha_u8, previous_alpha_u8)
        alpha_u8 = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        alpha_u8 = self._remove_isolated_foreground(alpha_u8, previous_alpha_u8)
        alpha_u8 = cv2.GaussianBlur(alpha_u8, (0, 0), sigmaX=0.6, sigmaY=0.6)

        if previous_alpha_u8 is not None:
            stable_prev = previous_alpha_u8.astype(np.float32)
            current = alpha_u8.astype(np.float32)
            recovered = np.where(
                (stable_prev > 200) & (current < 72),
                stable_prev * 0.12 + current * 0.88,
                current,
            )
            alpha_u8 = np.clip(recovered, 0, 255).astype(np.uint8)

        return alpha_u8.astype(np.float32) / 255.0

    def matte_image(self, image_bytes, previous_alpha=None):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_rgb = np.array(image)
        data = self.preprocess(image_rgb)
        img_input = np.expand_dims(data["img"], axis=0).astype("float32")

        self.input_handle.reshape(img_input.shape)
        self.input_handle.copy_from_cpu(img_input)
        self.predictor.run()

        alpha = self.output_handle.copy_to_cpu().squeeze(1)[0]
        alpha = self.postprocess(alpha, data["trans_info"])
        alpha = self.refine_alpha(alpha, previous_alpha=previous_alpha)
        alpha_image = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")

        cutout = image.copy()
        cutout.putalpha(alpha_image)

        composite = Image.alpha_composite(
            Image.new("RGBA", image.size, BACKGROUND_RGBA),
            cutout,
        ).convert("RGB")

        return {
            "alphaImage": encode_image(alpha_image, "PNG"),
            "cutoutImage": encode_image(cutout, "PNG"),
            "compositeImage": encode_image(composite, "PNG"),
            "summary": {
                "alpha": matte_summary(alpha_image),
            },
            "sourceSize": {"width": image.width, "height": image.height},
            "engine": "ppmattingv2",
            "alphaArray": alpha,
        }

    def process_directory(self, input_dir, output_dir):
        in_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = sorted(path for path in in_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not frame_paths:
            raise ValueError("No source frames found in the input directory.")

        started_at = time.perf_counter()
        preview = None
        previous_alpha = None

        for index, frame_path in enumerate(frame_paths):
            result = self.matte_image(frame_path.read_bytes(), previous_alpha=previous_alpha)
            previous_alpha = result.pop("alphaArray", None)
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


SERVICE = PPMattingV2Service()


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
        result = SERVICE.matte_image(image_bytes)
        result.pop("alphaArray", None)
        return {"id": request_id, **result}

    if action == "process_directory":
        return {"id": request_id, **SERVICE.process_directory(payload["inputDir"], payload["outputDir"])}

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
