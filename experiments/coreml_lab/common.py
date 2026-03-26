import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
RVM_ROOT = ROOT / "vendor" / "RobustVideoMatting"
MODEL_PATH = ROOT / "models" / "rvm_mobilenetv3.pth"

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
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            x = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            return x.reshape(b, t, c, target_h, target_w)

        return F.interpolate(
            x,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

    model._interpolate = _interpolate_with_integer_size
    return model


def patch_decoder_for_static_sizes(model: nn.Module, input_height: int, input_width: int):
    s1 = ((input_height + 1) // 2, (input_width + 1) // 2)
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
        x = F.interpolate(x, size=(input_height, input_width), mode="bilinear", align_corners=False)
        x = torch.cat([x, s], dim=1)
        x = decode0.conv(x)
        return x

    decode3.forward_single_frame = make_upsampling_forward(decode3, s3)
    decode2.forward_single_frame = make_upsampling_forward(decode2, s2)
    decode1.forward_single_frame = make_upsampling_forward(decode1, s1)
    decode0.forward_single_frame = _output_forward_single_frame
    return model


def build_wrapper(height: int, width: int, downsample_ratio: float, mode: str):
    model = load_rvm_model()
    model = patch_interpolate_for_conversion(
        model,
        input_height=height,
        input_width=width,
        downsample_ratio=downsample_ratio,
    )
    model = patch_decoder_for_static_sizes(model, input_height=height, input_width=width)
    return RVMSingleFrameWrapper(model, downsample_ratio=downsample_ratio, mode=mode).eval()


def make_test_input(height: int, width: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    base = np.zeros((1, 1, 3, height, width), dtype=np.float32)
    yy, xx = np.meshgrid(
        np.linspace(0, 1, height, dtype=np.float32),
        np.linspace(0, 1, width, dtype=np.float32),
        indexing="ij",
    )
    base[0, 0, 0] = xx
    base[0, 0, 1] = yy
    base[0, 0, 2] = 0.5 * (xx + yy)
    noise = rng.normal(0, 0.02, size=base.shape).astype(np.float32)
    return np.clip(base + noise, 0.0, 1.0)


def save_image(array: np.ndarray, path: Path):
    array = np.clip(array, 0.0, 1.0)
    if array.ndim == 2:
        image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    else:
        image = Image.fromarray((array * 255).astype(np.uint8), mode="RGB")
    image.save(path)
