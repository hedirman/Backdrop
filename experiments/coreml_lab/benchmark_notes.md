# Benchmark Notes

Latest sandbox benchmark at `128x128` single-frame input:

- `segmentation`: torch CPU `0.1334s`, Core ML `0.00443s`, about `30.1x` faster
- `alpha`: torch CPU `0.1327s`, Core ML `0.00443s`, about `29.9x` faster
- `matting_lite`: torch CPU `0.1312s`, Core ML `0.00450s`, about `29.2x` faster

Validation note:

- `matting_lite` numerically matches torch closely on the current synthetic input
- that synthetic input currently yields an all-zero alpha output in both paths
- next benchmark pass should use a real portrait frame so quality can be judged, not just speed

Use this file to continue comparing:

- current PyTorch CPU worker
- current PyTorch MPS worker if it ever becomes available
- current Core ML sandbox prototype

Suggested metrics:

- single-frame latency
- frames per second on preview-sized inputs
- frames per second on export-sized inputs
- memory usage
- qualitative edge quality
