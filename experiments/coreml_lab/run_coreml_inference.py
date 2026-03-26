import argparse
import json
import os
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from common import build_wrapper, make_test_input, save_image


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def parse_args():
    parser = argparse.ArgumentParser(description="Run sandbox Core ML inference and compare to PyTorch.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=["segmentation", "alpha", "matting_lite"], required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--downsample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def summarize_array(value: np.ndarray):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
    }


def main():
    args = parse_args()
    out_dir = ARTIFACTS / f"validation_{args.mode}_{args.height}x{args.width}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = HERE / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)
    tempfile.tempdir = str(tmpdir)

    x = make_test_input(args.height, args.width, seed=args.seed)
    np.save(out_dir / "input.npy", x)
    save_image(x[0, 0].transpose(1, 2, 0), out_dir / "input.png")

    model = ct.models.MLModel(args.model_path, compute_units=ct.ComputeUnit.ALL)
    coreml_outputs = model.predict({"src": x.astype(np.float32)})

    wrapper = build_wrapper(args.height, args.width, args.downsample_ratio, args.mode)
    with torch.inference_mode():
        torch_outputs = wrapper(torch.from_numpy(x))

    if args.mode == "segmentation":
        torch_outputs = {"segmentation": torch_outputs.numpy()}
    elif args.mode == "alpha":
        torch_outputs = {"alpha": torch_outputs.numpy()}
    else:
        foreground_residual, alpha = torch_outputs
        torch_outputs = {
            "foreground_residual": foreground_residual.numpy(),
            "alpha": alpha.numpy(),
        }

    report = {
        "mode": args.mode,
        "model_path": args.model_path,
        "coreml": {},
        "torch": {},
        "diff": {},
    }

    for name, value in coreml_outputs.items():
        value = np.asarray(value)
        coreml_outputs[name] = value
        report["coreml"][name] = summarize_array(value)
        np.save(out_dir / f"{name}_coreml.npy", value)

    for name, value in torch_outputs.items():
        value = np.asarray(value)
        torch_outputs[name] = value
        report["torch"][name] = summarize_array(value)
        np.save(out_dir / f"{name}_torch.npy", value)

    for name in coreml_outputs:
        if name not in torch_outputs:
            continue
        diff = np.abs(coreml_outputs[name] - torch_outputs[name])
        report["diff"][name] = {
            "mae": float(diff.mean()),
            "max_abs": float(diff.max()),
        }

    if "alpha" in coreml_outputs:
        save_image(coreml_outputs["alpha"][0, 0], out_dir / "alpha_coreml.png")
    if "alpha" in torch_outputs:
        save_image(torch_outputs["alpha"][0, 0], out_dir / "alpha_torch.png")
    if "foreground_residual" in coreml_outputs and "alpha" in coreml_outputs:
        coreml_cutout = coreml_outputs["foreground_residual"][0].transpose(1, 2, 0) * coreml_outputs["alpha"][0, 0][..., None]
        save_image(coreml_outputs["foreground_residual"][0].transpose(1, 2, 0), out_dir / "foreground_residual_coreml.png")
        save_image(coreml_cutout, out_dir / "cutout_coreml.png")
    if "foreground_residual" in torch_outputs and "alpha" in torch_outputs:
        torch_cutout = torch_outputs["foreground_residual"][0].transpose(1, 2, 0) * torch_outputs["alpha"][0, 0][..., None]
        save_image(torch_outputs["foreground_residual"][0].transpose(1, 2, 0), out_dir / "foreground_residual_torch.png")
        save_image(torch_cutout, out_dir / "cutout_torch.png")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
