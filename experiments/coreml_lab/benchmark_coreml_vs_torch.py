import json
import os
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from common import build_wrapper, make_test_input


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

CONFIGS = [
    ("segmentation", ARTIFACTS / "rvm_single_frame_segmentation_128x128.mlpackage"),
    ("alpha", ARTIFACTS / "rvm_single_frame_alpha_128x128.mlpackage"),
    ("matting_lite", ARTIFACTS / "rvm_single_frame_matting_lite_128x128.mlpackage"),
]


def benchmark_torch(wrapper, x, warmup=3, runs=10):
    tensor = torch.from_numpy(x)
    with torch.inference_mode():
        for _ in range(warmup):
            wrapper(tensor)
        start = time.perf_counter()
        for _ in range(runs):
            wrapper(tensor)
        end = time.perf_counter()
    return (end - start) / runs


def benchmark_coreml(model, x, warmup=3, runs=10):
    for _ in range(warmup):
        model.predict({"src": x})
    start = time.perf_counter()
    for _ in range(runs):
        model.predict({"src": x})
    end = time.perf_counter()
    return (end - start) / runs


def main():
    tmpdir = HERE / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)
    tempfile.tempdir = str(tmpdir)

    x = make_test_input(128, 128, seed=7).astype(np.float32)
    report = {
        "input_shape": list(x.shape),
        "results": [],
    }

    for mode, package_path in CONFIGS:
        wrapper = build_wrapper(128, 128, 1.0, mode)
        mlmodel = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.ALL)

        torch_s = benchmark_torch(wrapper, x)
        coreml_s = benchmark_coreml(mlmodel, x)

        report["results"].append(
            {
                "mode": mode,
                "torch_cpu_seconds": torch_s,
                "coreml_seconds": coreml_s,
                "speedup_vs_torch": torch_s / coreml_s if coreml_s > 0 else None,
            }
        )

    output_path = ARTIFACTS / "benchmark_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
