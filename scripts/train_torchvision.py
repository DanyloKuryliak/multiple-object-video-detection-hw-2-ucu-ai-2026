"""Fine-tune Faster R-CNN or RetinaNet on sequence-separated COCO data."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from detection import VideoCoco, build_model


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = VideoCoco(args.train_dir, args.stride, -1 if args.family == "retinanet" else 0)
    assert dataset.images, "No training images found"
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
    model = build_model(args.family, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch={epoch} mean_train_loss={total / len(loader):.5f}", flush=True)
        torch.save({"family": args.family, "epoch": epoch, "seed": args.seed, "stride": args.stride,
                    "model": model.state_dict()}, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["fasterrcnn", "retinanet"], required=True)
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--stride", type=int, default=10, help="Sample every Nth frame per clip")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    assert args.epochs > 0 and args.stride > 0 and args.batch_size > 0 and args.lr > 0
    train(args)
