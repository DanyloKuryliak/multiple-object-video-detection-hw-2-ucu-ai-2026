"""Run consistent COCO validation or write a Kaggle submission."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detection import build_model, image_tensor


def predictor(family, checkpoint, device):
    if family in {"fasterrcnn", "retinanet"}:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["family"] == family, "Checkpoint family mismatch"
        model = build_model(family, pretrained=False)
        model.load_state_dict(saved["model"])
        model.to(device).eval()

        def detect(path):
            with torch.inference_mode():
                result = model([image_tensor(path).to(device)])[0]
            labels = result["labels"] + (1 if family == "retinanet" else 0)
            return zip(labels.tolist(), result["scores"].tolist(), result["boxes"].tolist())

    else:
        from ultralytics import RTDETR, YOLO

        model = (RTDETR if family == "rtdetr" else YOLO)(str(checkpoint))

        def detect(path):
            with Image.open(path) as image:
                result = model.predict(source=image.convert("RGB"), conf=0.001, max_det=500,
                                       device=device, verbose=False)[0]
            return zip((result.boxes.cls.int() + 1).tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())

    return detect


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def run(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    detect = predictor(args.family, args.checkpoint, device)
    if args.split == "val":
        root = args.data / "val"
        coco_file = root / "annotations_coco.json"
        images = json.loads(coco_file.read_text())["images"]
    else:
        root = args.data
        with (root / "test_images.csv").open(newline="") as source:
            images = list(csv.DictReader(source))
    detections, rows = [], []
    started = time.perf_counter()
    for index, image in enumerate(images, 1):
        width, height = int(image["width"]), int(image["height"])
        parts = []
        for label, score, (x1, y1, x2, y2) in detect(root / image["file_name"]):
            x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
            y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
            if not (1 <= label <= 8 and x2 > x1 and y2 > y1):
                continue
            if args.split == "val":
                detections.append({"image_id": image["id"], "category_id": label,
                                   "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
            else:
                parts.extend((str(label - 1), f"{score:.5f}", f"{x1:.1f}", f"{y1:.1f}",
                              f"{x2 - x1:.1f}", f"{y2 - y1:.1f}"))
            if len(parts) >= 500 * 6:
                break
        if args.split == "test":
            rows.append((image["image_id"], " ".join(parts)))
        if index % 100 == 0:
            print(f"processed {index}/{len(images)}", flush=True)
    sync(device)
    seconds = time.perf_counter() - started
    if args.split == "test":
        assert len(rows) == len(images) and len({row[0] for row in rows}) == len(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(("image_id", "PredictionString"))
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.output}")
    else:
        scores = {"mAP": 0.0, "AP50": 0.0, "AP_small": 0.0}
        if detections:
            coco = COCO(str(coco_file))
            evaluation = COCOeval(coco, coco.loadRes(detections), "bbox")
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()
            scores = {"mAP": float(evaluation.stats[0]), "AP50": float(evaluation.stats[1]),
                      "AP_small": float(evaluation.stats[3])}
        metrics = {**scores, "images": len(images), "seconds_per_image": seconds / len(images),
                   "family": args.family, "checkpoint": str(args.checkpoint)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["fasterrcnn", "retinanet", "yolo", "rtdetr"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    run(parser.parse_args())
