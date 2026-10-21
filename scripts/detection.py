"""Shared COCO loader and TorchVision model setup for the two anchor-based baselines."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms.functional import pil_to_tensor


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float().div_(255)


class VideoCoco(Dataset):
    def __init__(self, split: Path, stride: int = 1, label_offset: int = 0):
        self.split = split
        self.label_offset = label_offset
        coco = json.loads((split / "annotations_coco.json").read_text())
        self.images = [image for image in coco["images"] if (int(image["frame_id"]) - 1) % stride == 0]
        self.annotations = {image["id"]: [] for image in self.images}
        for annotation in coco["annotations"]:
            if annotation["image_id"] in self.annotations and not annotation.get("iscrowd", 0):
                self.annotations[annotation["image_id"]].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        boxes, labels = [], []
        for annotation in self.annotations[image["id"]]:
            x, y, width, height = annotation["bbox"]
            if width > 0 and height > 0:
                boxes.append([x, y, x + width, y + height])
                labels.append(annotation["category_id"] + self.label_offset)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": image["id"],
        }
        return image_tensor(self.split / image["file_name"]), target


def build_model(family: str, pretrained: bool):
    if family == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None, weights_backbone=None)
        channels = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(channels, 9)
    elif family == "retinanet":
        model = retinanet_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None, weights_backbone=None)
        anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(256, anchors, 8)
    else:
        raise ValueError(f"Unknown model family: {family}")
    return model
