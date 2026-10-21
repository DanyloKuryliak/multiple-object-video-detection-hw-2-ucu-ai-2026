"""Inspect supplied COCO annotations before training."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
import yaml


def inspect(split: Path) -> dict:
    coco = json.loads((split / "annotations_coco.json").read_text())
    images = coco["images"]
    by_id = {image["id"]: image for image in images}
    classes = Counter()
    clips = defaultdict(lambda: {"frames": 0, "objects": 0})
    sizes = Counter()
    density = Counter()
    targets = defaultdict(list)
    for image in images:
        clips[image["video"]]["frames"] += 1
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        image = by_id[annotation["image_id"]]
        targets[image["id"]].append(annotation)
        classes[annotation["category_id"]] += 1
        clips[image["video"]]["objects"] += 1
        area = annotation["bbox"][2] * annotation["bbox"][3]
        sizes["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
    for image in images:
        density[len(targets[image["id"]])] += 1
    sample = next((image for image in images if targets[image["id"]]), None)
    if sample:
        source = split / sample["file_name"]
        with Image.open(source) as original:
            preview = original.convert("RGB")
        draw = ImageDraw.Draw(preview)
        for annotation in targets[sample["id"]]:
            x, y, w, h = annotation["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline="red", width=2)
        preview.save(split / "annotation_preview.jpg")
    return {
        "frames": len(images),
        "clips": dict(sorted(clips.items())),
        "classes_1_based": dict(sorted(classes.items())),
        "object_sizes_coco": dict(sizes),
        "objects_per_frame": dict(sorted(density.items())),
        "ignored_regions": sum(bool(a.get("iscrowd", 0)) for a in coco["annotations"]),
        "preview": str(split / "annotation_preview.jpg") if sample else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    train, val = inspect(args.data / "train"), inspect(args.data / "val")
    train_clips, val_clips = set(train["clips"]), set(val["clips"])
    assert not train_clips & val_clips, "train and val share a video sequence"
    names = yaml.safe_load((args.data / "train" / "data.yaml").read_text())["names"]
    assert names == yaml.safe_load((args.data / "val" / "data.yaml").read_text())["names"]
    (args.data / "detection.yaml").write_text(yaml.safe_dump({
        "path": str(args.data.resolve()), "train": "train/images", "val": "val/images",
        "test": "test/images", "nc": len(names), "names": names,
    }, sort_keys=False))
    result = {"train": train, "val": val}
    (args.data / "eda.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
