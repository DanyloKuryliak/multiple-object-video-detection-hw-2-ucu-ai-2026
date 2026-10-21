"""Create sequence-level folds shared by TorchVision and Ultralytics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


def prepare(data: Path, count: int):
    sources = {}
    videos = Counter()
    for split in ("train", "val"):
        sources[split] = json.loads((data / split / "annotations_coco.json").read_text())
        videos.update(image["video"] for image in sources[split]["images"])
    assert len(videos) >= count >= 2
    assert sum(videos.values()) == sum(len(source["images"]) for source in sources.values())
    assigned = {}
    sizes = [0] * count
    for video, frames in sorted(videos.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(count), key=lambda index: (sizes[index], index))
        assigned[video] = fold
        sizes[fold] += frames
    names = yaml.safe_load((data / "train" / "data.yaml").read_text())["names"]
    for fold in range(count):
        root = data / "folds" / f"fold_{fold}"
        for destination in ("train", "val"):
            output = root / destination
            for subdir in ("images", "labels"):
                (output / subdir).mkdir(parents=True, exist_ok=True)
            images, annotations = [], []
            image_ids = {}
            for source_name, source in sources.items():
                source_root = data / source_name
                for image in source["images"]:
                    selected = "val" if assigned[image["video"]] == fold else "train"
                    if selected != destination:
                        continue
                    old_id = image["id"]
                    new_id = len(images) + 1
                    image_ids[(source_name, old_id)] = new_id
                    images.append({**image, "id": new_id})
                    for subdir in ("images", "labels"):
                        link = output / subdir / image["video"]
                        if not link.exists():
                            link.symlink_to((source_root / subdir / image["video"]).resolve(), target_is_directory=True)
                for annotation in source["annotations"]:
                    new_id = image_ids.get((source_name, annotation["image_id"]))
                    if new_id is not None:
                        annotations.append({**annotation, "id": len(annotations) + 1, "image_id": new_id})
            assert images and annotations
            (output / "annotations_coco.json").write_text(json.dumps({
                "images": images, "annotations": annotations, "categories": sources["train"]["categories"],
            }))
        (root / "detection.yaml").write_text(yaml.safe_dump({
            "path": str(root.resolve()), "train": "train/images", "val": "val/images",
            "nc": len(names), "names": names,
        }, sort_keys=False))
        print(f"fold {fold}: {sizes[fold]} validation frames")
    (data / "folds" / "assignment.json").write_text(json.dumps(assigned, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    prepare(args.data, args.folds)
