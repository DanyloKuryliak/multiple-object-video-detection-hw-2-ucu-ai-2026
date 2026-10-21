# Video Object Detection in Short Clips

This project studies object detection in video frames for the UCU AI course homework. The goal is to detect the annotated object classes in each frame and generate predictions in the Kaggle competition format.

## Dataset

The data is provided through the [Kaggle competition](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026). It is not stored in this repository. Training can use Kaggle Notebooks, where the competition data can be attached directly to the runtime.

## Code

- `scripts/inspect_data.py` inspects the dataset and prepares the detector configuration.
- `scripts/prepare_folds.py` creates sequence-level cross-validation folds.
- `scripts/train_torchvision.py` trains Faster R-CNN or RetinaNet models.
- `scripts/predict.py` evaluates models and creates competition-format predictions.
- YOLO and RT-DETR training use the Ultralytics CLI.
- `kaggle/` contains the Kaggle Notebook runner and its metadata. The current runner checks the mounted data and GPU; it does not yet start model training.

The environment is defined by `pyproject.toml` and `uv.lock`. To install it locally, run:

```bash
uv sync --locked
```

Training and validation results are recorded in [`SUBMISSIONS.md`](SUBMISSIONS.md). No experiment results have been recorded yet. Model weights, datasets, and generated predictions are excluded from version control.

## References

- [Kaggle competition](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026)
- [TorchVision object detection models](https://pytorch.org/vision/stable/models.html#object-detection)
- [Ultralytics object detection](https://docs.ultralytics.com/tasks/detect/)
- [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/)

## Report

The course report is maintained in [`report.md`](report.md) and published through GitHub Pages.
