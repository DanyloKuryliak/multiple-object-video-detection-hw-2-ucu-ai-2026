"""Kaggle smoke runner; later experiments run the versioned scripts from GitHub."""

import subprocess
from pathlib import Path


SLUG = "multiple-object-video-detection-hw-2-ucu-ai-2026"
REPO = "https://github.com/DanyloKuryliak/multiple-object-video-detection-hw-2-ucu-ai-2026.git"
workspace = Path("/kaggle/working/hw2")
subprocess.run(["git", "clone", "--depth", "1", REPO, str(workspace)], check=True)
data = next((path for path in (
    Path("/kaggle/input") / SLUG,
    Path("/kaggle/input/competitions") / SLUG,
) if (path / "train/annotations_coco.json").is_file()), None)
assert data is not None, "Competition data is not mounted in Kaggle Inputs"
(workspace / "data").symlink_to(data, target_is_directory=True)
import torch

print("repo:", subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "--short", "HEAD"], text=True).strip())
print("data:", data)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("ready to run scripts from", workspace)
