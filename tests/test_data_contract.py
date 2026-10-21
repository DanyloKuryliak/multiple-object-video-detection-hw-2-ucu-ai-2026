import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from detection import VideoCoco
import predict


class DataContractTest(unittest.TestCase):
    def test_class_offsets_and_ignored_regions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images" / "clip_a").mkdir(parents=True)
            Image.new("RGB", (20, 20)).save(root / "images" / "clip_a" / "1.jpg")
            (root / "annotations_coco.json").write_text(json.dumps({
                "images": [{"id": 1, "file_name": "images/clip_a/1.jpg", "frame_id": 1}],
                "annotations": [
                    {"image_id": 1, "category_id": 8, "bbox": [1, 2, 3, 4], "iscrowd": 0},
                    {"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 1},
                ],
            }))
            _, faster = VideoCoco(root)[0]
            _, retina = VideoCoco(root, label_offset=-1)[0]
            self.assertEqual(faster["labels"].tolist(), [8])
            self.assertEqual(retina["labels"].tolist(), [7])
            self.assertEqual(faster["boxes"].tolist(), [[1.0, 2.0, 4.0, 6.0]])

    def test_empty_validation_predictions_score_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "val").mkdir()
            (root / "val" / "annotations_coco.json").write_text(json.dumps({
                "images": [{"id": 1, "file_name": "unused.jpg", "width": 20, "height": 20}],
                "annotations": [], "categories": [{"id": 1, "name": "person"}],
            }))
            output = root / "metrics.json"
            args = Namespace(family="yolo", checkpoint=Path("unused.pt"), split="val", data=root,
                             output=output, device="cpu")
            with patch.object(predict, "predictor", return_value=lambda _: []):
                predict.run(args)
            self.assertEqual(json.loads(output.read_text())["mAP"], 0.0)


if __name__ == "__main__":
    unittest.main()
