# Core ML Lab

This folder is an isolated sandbox for Apple-native acceleration experiments.

Its purpose is to explore whether `Backdrop` can move video matting inference from PyTorch to a Core ML-based path without affecting the current production app flow.

## Isolation Rules

- Do not import this folder from `server.js`
- Do not point the live UI to anything in this folder
- Do not replace the current RVM worker from here
- Treat this as a scratchpad for conversion, validation, and benchmarking only

## Current Goal

1. Verify whether the existing RVM model can be exported or adapted to Core ML
2. Measure whether Core ML offers a practical acceleration path on this machine
3. Decide whether a future native worker should be built in Swift or via Python conversion tooling

## Suggested Workflow

1. Create a conversion script for a minimal RVM export attempt
2. Run a tiny smoke test on a single frame
3. Compare:
   - conversion success
   - inference correctness
   - speed vs current PyTorch worker

## Notes

- The live app currently uses `scripts/rvm_worker.py`
- The live app currently uses the standard Node server in the repo root
- Nothing in this folder is wired into the app yet

## First Conversion Attempt

The current first-pass experiment is a single-frame wrapper around the existing MobileNetV3 RVM model.

Run it with:

```bash
./.venv314/bin/python experiments/coreml_lab/convert_rvm_to_coreml.py --height 256 --width 256
```

Expected output:

- a `.mlpackage` in `experiments/coreml_lab/artifacts`
- a small JSON metadata file beside it

This experiment is intentionally limited:

- no live app integration
- no recurrent-state export yet
- no production worker replacement yet

## Current Findings

First sandbox conversion attempt status:

- `coremltools` installed successfully in the isolated Python 3.14 environment
- a single-frame RVM wrapper can be traced in the sandbox
- the first conversion blocker was RVM's float resize path
- after patching that in the sandbox, conversion progressed further
- the next blocker is a Core ML conversion failure around an `int` op in the traced graph
- moving the sandbox to Python 3.12 with a native `coremltools` wheel removed the missing native-library blocker
- a reduced single-frame segmentation-only RVM wrapper now converts successfully to Core ML
- a single-frame alpha-only wrapper now converts successfully to Core ML
- a single-frame lightweight foreground-residual + alpha wrapper now converts successfully to Core ML

What this means:

- the Core ML direction is plausible enough to keep exploring
- the current full RVM matting graph is not plug-and-play for Core ML conversion
- smaller Core ML-compatible slices of RVM are now proven
- a lightweight foreground + alpha output path is now proven
- the remaining gap is bridging from `matting_lite` to the full production RVM output path

## Validation And Benchmark Status

The sandbox now includes three supporting scripts:

- `convert_rvm_to_coreml.py`
- `run_coreml_inference.py`
- `benchmark_coreml_vs_torch.py`

### Inference Validation

`run_coreml_inference.py` was run successfully against:

- `artifacts/rvm_single_frame_matting_lite_128x128.mlpackage`

What was validated:

- Core ML model loading works in the sandbox Python 3.12 environment
- `predict()` works when `TMPDIR` is forced into the sandbox
- output tensors match the torch wrapper very closely on the synthetic test input

Observed `matting_lite` comparison on the current synthetic input:

- `foreground_residual` mean absolute error: about `0.00049`
- `foreground_residual` max absolute error: about `0.01248`
- `alpha` matched exactly on this input

Important caveat:

- the current deterministic synthetic input produces an all-zero alpha result in both torch and Core ML
- that means numerical agreement is proven, but visual usefulness is not yet proven
- the next validation pass should use a real portrait or person-like frame

### Benchmark Results

`benchmark_coreml_vs_torch.py` was run successfully at `128x128` with the single-frame wrappers.

Measured average per-run latency:

- `segmentation`
  - torch CPU: `0.1334s`
  - Core ML: `0.00443s`
  - speedup: about `30.1x`
- `alpha`
  - torch CPU: `0.1327s`
  - Core ML: `0.00443s`
  - speedup: about `29.9x`
- `matting_lite`
  - torch CPU: `0.1312s`
  - Core ML: `0.00450s`
  - speedup: about `29.2x`

What this means:

- Apple-native inference is no longer theoretical in this repo
- Core ML is already dramatically faster than the current torch CPU sandbox path for these reduced single-frame exports
- the open question is now output quality and whether `matting_lite` is visually good enough for preview use

## Recommended Next Steps

1. Run `run_coreml_inference.py` on a real portrait frame instead of the synthetic gradient input.
2. Inspect the generated `alpha`, `foreground_residual`, and `cutout` artifacts visually.
3. If quality looks usable, build a tiny isolated Core ML worker in this sandbox.
4. Keep the current PyTorch RVM worker as fallback until the Core ML path proves stable on real footage.

## Browser Sandbox

There is now a separate lab server in this folder for manual testing:

- `server.js`
- `coreml_worker.py`
- `public/`

It is intentionally isolated from the main app and currently supports:

- image testing
- sandbox video testing

What it does:

- loads the `matting_lite` Core ML package
- accepts a single uploaded image or a single uploaded video
- returns:
  - alpha
  - foreground residual
  - cutout preview
  - for video, an encoded preview MP4 built from processed frames

This is meant to answer two questions before deeper integration:

- does the current Core ML sandbox output look visually usable on real images?
- does the current Core ML sandbox output stay usable enough across video frames to justify a preview-worker experiment?
