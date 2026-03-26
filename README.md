# Backdrop

Backdrop is a local multi-engine web app for video background removal, replacement, preview, and MP4 export.

It supports multiple processing engines:

- `Core ML Preview`
- `Core ML Quality`
- `PP-MattingV2`
- `MODNet`
- `RVM`

## Features

- Upload a local video
- Remove and replace the background
- Switch between multiple matting/segmentation engines
- Tune mask and compositing controls
- Generate a local MP4 preview
- Export the processed result as MP4

## Project Structure

- `public/index.html`: main Backdrop UI
- `public/styles.css`: app styling
- `public/app.js`: frontend logic, controls, preview, and export flow
- `server.js`: local Node server, worker routing, processing, and export endpoints
- `scripts/rvm_worker.py`: Robust Video Matting worker
- `scripts/modnet_worker.py`: MODNet worker
- `scripts/ppmattingv2_worker.py`: PP-MattingV2 worker
- `experiments/coreml_lab/`: Core ML conversion, benchmarking, and research utilities

## Requirements

- Node.js 18+
- Python 3.12+ recommended
- `ffmpeg`
- macOS is recommended if you want to use the Core ML paths

## Installation

### 1. Clone the repo

```bash
git clone <your-private-repo-url>
cd Backdrop
```

### 2. Install system tools

On macOS with Homebrew:

```bash
brew install node ffmpeg python@3.12
```

### 3. Install Node dependencies

The app currently has no third-party npm dependencies, but running this is still fine:

```bash
npm install
```

### 4. Create the Python environments

Backdrop uses:

- a root environment for `RVM` and `MODNet`
- a Core ML lab environment for the Core ML worker and `PP-MattingV2`

#### Root worker environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision pillow numpy
deactivate
```

If you want to use the alternate root environment name used by the app:

```bash
python3 -m venv .venv314
source .venv314/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision pillow numpy
deactivate
```

#### Core ML / PP-MattingV2 environment

```bash
python3 -m venv experiments/coreml_lab/.venv312
source experiments/coreml_lab/.venv312/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision pillow numpy coremltools
pip install paddlepaddle opencv-python pyyaml six pymatting scikit-image numba
pip install requests tqdm filelock prettytable scikit-learn visualdl
deactivate
```

## Model Assets

This repo expects the model assets used by the workers to be present:

- `models/rvm_mobilenetv3.pth`
- `vendor/MODNet/pretrained/modnet_webcam_portrait_matting.ckpt`
- `models/ppmattingv2/ppmattingv2-stdc1-human_512/`
- `vendor/RobustVideoMatting/`
- `vendor/MODNet/`
- `vendor/PaddleSeg/`

If you are setting the project up from scratch, make sure those model and vendor files are included in the repo or downloaded before running the app.

## Core ML Path

The Core ML modes in Backdrop were transposed from the original PyTorch RVM workflow into Apple-native Core ML packages.

The Core ML worker and converted model artifacts live under:

- `experiments/coreml_lab/coreml_worker.py`
- `experiments/coreml_lab/artifacts/`

To run the Core ML engines successfully, you need:

- macOS
- the Core ML environment at `experiments/coreml_lab/.venv312`
- `coremltools` installed in that environment
- the generated Core ML model packages inside `experiments/coreml_lab/artifacts/`

If those pieces are missing, the non-Core-ML engines such as `RVM`, `MODNet`, and `PP-MattingV2` can still run independently.

## Run the App

```bash
npm start
```

Then choose an engine inside the app:

- `Core ML Preview` and `Core ML Quality` use the converted Core ML models
- `RVM`, `MODNet`, and `PP-MattingV2` use their respective Python worker paths

## Engine Notes

- `Core ML Preview`: fastest preview-oriented Core ML path
- `Core ML Quality`: stronger Core ML path
- `PP-MattingV2`: strong edge quality, good alternative engine
- `MODNet`: lightweight portrait matting engine
- `RVM`: strongest temporal-quality reference path

## Notes

- Performance depends heavily on your local machine, Python setup, and available acceleration.
- `ffmpeg` is required for preview and export.
