"""Verify that Kaggle has mounted the competition data and enabled a GPU."""

from pathlib import Path

import torch

SLUG = "multiple-object-video-detection-hw-2-ucu-ai-2026"
data = next((path for path in (
    Path("/kaggle/input") / SLUG,
    Path("/kaggle/input/competitions") / SLUG,
) if (path / "train/annotations_coco.json").is_file()), None)
assert data is not None, "Competition data is not mounted in Kaggle Inputs"
assert torch.cuda.is_available(), "Kaggle GPU is not enabled"

print("data:", data)
print("torch:", torch.__version__)
print("gpu:", torch.cuda.get_device_name(0))
print("competition data and GPU are ready")
