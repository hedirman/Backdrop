import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
RVM_ROOT = ROOT / "vendor" / "RobustVideoMatting"
MODEL_PATH = ROOT / "models" / "rvm_mobilenetv3.pth"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

sys.path.insert(0, str(RVM_ROOT))

from model import MattingNetwork  # noqa: E402


class RVMSingleFrameWrapper(nn.Module):
    def __init__(self, model: nn.Module, downsample_ratio: float, mode: str = "matting"):
        super().__init__()
        self.model = model
        self.downsample_ratio = float(downsample_ratio)
        self.mode = mode

    def forward(self, src: torch.Tensor):
        if src.ndim == 4:
            src = src.unsqueeze(1)

        if self.mode == "segmentation":
            outputs = self.model(
                src,
                downsample_ratio=self.downsample_ratio,
                segmentation_pass=True,
            )
            seg, *_ = outputs
            return seg.squeeze(1)

        if self.mode == "alpha":
            src_sm = src
            f1, f2, f3, f4 = self.model.backbone(src_sm)
            f4 = self.model.aspp(f4)
            hid, *_ = self.model.decoder(src_sm, f1, f2, f3, f4, None, None, None, None)
            mat = self.model.project_mat(hid)
            pha = mat[:, :, 3:4]
            pha = pha.clamp(0.0, 1.0)
            return pha.squeeze(1)

        if self.mode == "matting_lite":
            src_sm = src
            f1, f2, f3, f4 = self.model.backbone(src_sm)
            f4 = self.model.aspp(f4)
            hid, *_ = self.model.decoder(src_sm, f1, f2, f3, f4, None, None, None, None)
            mat = self.model.project_mat(hid)
            fgr_residual = mat[:, :, :3]
            pha = mat[:, :, 3:4]
            fgr_residual = fgr_residual.clamp(0.0, 1.0)
            pha = pha.clamp(0.0, 1.0)
            return fgr_residual.squeeze(1), pha.squeeze(1)

        outputs = self.model(
            src,
            downsample_ratio=self.downsample_ratio,
            segmentation_pass=False,
        )
        fgr, pha, *_ = outputs
        return fgr.squeeze(1), pha.squeeze(1)


class RVMRecurrentFrameWrapper(nn.Module):
    def __init__(self, model: nn.Module, downsample_ratio: float):
        super().__init__()
        self.model = model
        self.downsample_ratio = float(downsample_ratio)

    def forward(
        self,
        src: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
        r3: torch.Tensor,
        r4: torch.Tensor,
    ):
        fgr, pha, nr1, nr2, nr3, nr4 = self.model(
            src,
            r1,
            r2,
            r3,
            r4,
            downsample_ratio=self.downsample_ratio,
            segmentation_pass=False,
        )
        return fgr, pha, nr1, nr2, nr3, nr4


def load_rvm_model():
    model = MattingNetwork("mobilenetv3", pretrained_backbone=False).eval()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def patch_interpolate_for_conversion(model: nn.Module, input_height: int, input_width: int, downsample_ratio: float):
    target_h = max(1, round(input_height * downsample_ratio))
    target_w = max(1, round(input_width * downsample_ratio))

    def _interpolate_with_integer_size(x: torch.Tensor, scale_factor: float):
        if x.ndim == 5:
            # The conversion sandbox always traces with a fixed single-frame
            # 5D input of shape (1, 1, C, H, W). Avoid dynamic shape->int
            # conversions here so Core ML doesn't see tensor-to-int casts.
            x = x[:, 0]
            x = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            return x.unsqueeze(1)

        return F.interpolate(
            x,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

    model._interpolate = _interpolate_with_integer_size
    return model


def patch_decoder_for_static_sizes(model: nn.Module, input_height: int, input_width: int, downsample_ratio: float):
    base_height = max(1, round(input_height * downsample_ratio))
    base_width = max(1, round(input_width * downsample_ratio))
    s1 = ((base_height + 1) // 2, (base_width + 1) // 2)
    s2 = ((s1[0] + 1) // 2, (s1[1] + 1) // 2)
    s3 = ((s2[0] + 1) // 2, (s2[1] + 1) // 2)

    decode3 = model.decoder.decode3
    decode2 = model.decoder.decode2
    decode1 = model.decoder.decode1
    decode0 = model.decoder.decode0

    def make_upsampling_forward(block, target_hw):
        def _forward_single_frame(x, f, s, r):
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            x = torch.cat([x, f, s], dim=1)
            x = block.conv(x)
            a, b = x.split(block.out_channels // 2, dim=1)
            b, r = block.gru(b, r)
            x = torch.cat([a, b], dim=1)
            return x, r

        return _forward_single_frame

    def _output_forward_single_frame(x, s):
        x = F.interpolate(x, size=(base_height, base_width), mode="bilinear", align_corners=False)
        x = torch.cat([x, s], dim=1)
        x = decode0.conv(x)
        return x

    decode3.forward_single_frame = make_upsampling_forward(decode3, s3)
    decode2.forward_single_frame = make_upsampling_forward(decode2, s2)
    decode1.forward_single_frame = make_upsampling_forward(decode1, s1)
    decode0.forward_single_frame = _output_forward_single_frame
    return model


def patch_refiner_for_static_sizes(model: nn.Module, input_height: int, input_width: int):
    refiner = model.refiner

    if hasattr(refiner, "box_filter") and hasattr(refiner, "conv"):
        def _forward_single_frame(fine_src, base_src, base_fgr, base_pha, base_hid):
            fine_x = torch.cat([fine_src, fine_src.mean(1, keepdim=True)], dim=1)
            base_x = torch.cat([base_src, base_src.mean(1, keepdim=True)], dim=1)
            base_y = torch.cat([base_fgr, base_pha], dim=1)

            mean_x = refiner.box_filter(base_x)
            mean_y = refiner.box_filter(base_y)
            cov_xy = refiner.box_filter(base_x * base_y) - mean_x * mean_y
            var_x = refiner.box_filter(base_x * base_x) - mean_x * mean_x

            A = refiner.conv(torch.cat([cov_xy, var_x, base_hid], dim=1))
            b = mean_y - A * mean_x

            A = F.interpolate(A, size=(input_height, input_width), mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=(input_height, input_width), mode="bilinear", align_corners=False)

            out = A * fine_x + b
            fgr, pha = out.split([3, 1], dim=1)
            return fgr, pha

        refiner.forward_single_frame = _forward_single_frame
        return model

    if hasattr(refiner, "guilded_filter"):
        guided_filter = refiner.guilded_filter

        def _guided_filter_forward(lr_x, lr_y, hr_x):
            mean_x = guided_filter.boxfilter(lr_x)
            mean_y = guided_filter.boxfilter(lr_y)
            cov_xy = guided_filter.boxfilter(lr_x * lr_y) - mean_x * mean_y
            var_x = guided_filter.boxfilter(lr_x * lr_x) - mean_x * mean_x
            A = cov_xy / (var_x + guided_filter.eps)
            b = mean_y - A * mean_x
            A = F.interpolate(A, size=(input_height, input_width), mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=(input_height, input_width), mode="bilinear", align_corners=False)
            return A * hr_x + b

        guided_filter.forward = _guided_filter_forward
        return model

    return model


def infer_recurrent_state_shapes(model: nn.Module, height: int, width: int, downsample_ratio: float):
    src = torch.rand(1, 3, height, width)
    with torch.inference_mode():
        _, _, r1, r2, r3, r4 = model(
            src,
            None,
            None,
            None,
            None,
            downsample_ratio=downsample_ratio,
            segmentation_pass=False,
        )
    return [tuple(state.shape) for state in (r1, r2, r3, r4)]


def convert_single_frame(height: int, width: int, downsample_ratio: float, mode: str):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_rvm_model()
    model = patch_interpolate_for_conversion(
        model,
        input_height=height,
        input_width=width,
        downsample_ratio=downsample_ratio,
    )
    model = patch_decoder_for_static_sizes(
        model,
        input_height=height,
        input_width=width,
        downsample_ratio=downsample_ratio,
    )
    model = patch_refiner_for_static_sizes(model, input_height=height, input_width=width)
    wrapper = RVMSingleFrameWrapper(
        model,
        downsample_ratio=downsample_ratio,
        mode=mode,
    ).eval()
    example = torch.rand(1, 1, 3, height, width)

    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced(example)

    if mode == "segmentation":
        output_types = [ct.TensorType(name="segmentation")]
    elif mode == "alpha":
        output_types = [ct.TensorType(name="alpha")]
    elif mode == "matting_lite":
        output_types = [
            ct.TensorType(name="foreground_residual"),
            ct.TensorType(name="alpha"),
        ]
    else:
        output_types = [
            ct.TensorType(name="foreground"),
            ct.TensorType(name="alpha"),
        ]
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,
        inputs=[
            ct.TensorType(name="src", shape=example.shape),
        ],
        outputs=output_types,
    )

    mode_name = mode
    package_path = ARTIFACTS_DIR / f"rvm_single_frame_{mode_name}_{height}x{width}.mlpackage"
    metadata_path = ARTIFACTS_DIR / f"rvm_single_frame_{mode_name}_{height}x{width}.json"
    mlmodel.save(str(package_path))

    metadata = {
        "mode": f"single_frame_{mode_name}",
        "height": height,
        "width": width,
        "downsample_ratio": downsample_ratio,
        "package_path": str(package_path),
        "torch_version": torch.__version__,
        "coremltools_version": ct.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))


def convert_recurrent_frame(height: int, width: int, downsample_ratio: float):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_rvm_model()
    model = patch_interpolate_for_conversion(
        model,
        input_height=height,
        input_width=width,
        downsample_ratio=downsample_ratio,
    )
    model = patch_decoder_for_static_sizes(
        model,
        input_height=height,
        input_width=width,
        downsample_ratio=downsample_ratio,
    )
    model = patch_refiner_for_static_sizes(model, input_height=height, input_width=width)

    state_shapes = infer_recurrent_state_shapes(model, height, width, downsample_ratio)
    wrapper = RVMRecurrentFrameWrapper(model, downsample_ratio=downsample_ratio).eval()

    example_src = torch.rand(1, 3, height, width)
    example_states = [torch.zeros(shape) for shape in state_shapes]

    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (example_src, *example_states), strict=False)
        traced(example_src, *example_states)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,
        inputs=[
            ct.TensorType(name="src", shape=example_src.shape),
            ct.TensorType(name="r1", shape=example_states[0].shape),
            ct.TensorType(name="r2", shape=example_states[1].shape),
            ct.TensorType(name="r3", shape=example_states[2].shape),
            ct.TensorType(name="r4", shape=example_states[3].shape),
        ],
        outputs=[
            ct.TensorType(name="foreground"),
            ct.TensorType(name="alpha"),
            ct.TensorType(name="nr1"),
            ct.TensorType(name="nr2"),
            ct.TensorType(name="nr3"),
            ct.TensorType(name="nr4"),
        ],
    )

    package_path = ARTIFACTS_DIR / f"rvm_recurrent_frame_matting_{height}x{width}.mlpackage"
    metadata_path = ARTIFACTS_DIR / f"rvm_recurrent_frame_matting_{height}x{width}.json"
    mlmodel.save(str(package_path))

    metadata = {
        "mode": "recurrent_frame_matting",
        "height": height,
        "width": width,
        "downsample_ratio": downsample_ratio,
        "state_shapes": state_shapes,
        "package_path": str(package_path),
        "torch_version": torch.__version__,
        "coremltools_version": ct.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Sandboxed RVM to Core ML conversion experiment.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--downsample-ratio", type=float, default=0.35)
    parser.add_argument(
        "--mode",
        choices=["segmentation", "alpha", "matting_lite", "matting", "recurrent_matting"],
        default="matting",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "recurrent_matting":
        convert_recurrent_frame(
            height=args.height,
            width=args.width,
            downsample_ratio=args.downsample_ratio,
        )
        return
    convert_single_frame(
        height=args.height,
        width=args.width,
        downsample_ratio=args.downsample_ratio,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
