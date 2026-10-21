"""Verify that Kaggle has mounted the competition data and enabled a GPU."""

from pathlib import Path

import shutil
import subprocess

import torch

SLUG = "multiple-object-video-detection-hw-2-ucu-ai-2026"
data = next((path for path in (
    Path("/kaggle/input") / SLUG,
    Path("/kaggle/input/competitions") / SLUG,
) if (path / "train/annotations_coco.json").is_file()), None)
assert data is not None, "Competition data is not mounted in Kaggle Inputs"

print("data:", data)
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("GPU unavailable; enable it in the Kaggle Notebook settings")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=False)
    else:
        print("nvidia-smi is not available in this runtime")
