"""
Fine-tune YOLO11n on the indoor-navigation dataset.

Thin, explicit wrapper around the Ultralytics trainer. Its job is reproducibility:
the exact hyperparameters that produced a checkpoint live in the repo, not in
shell history.

Defaults are tuned for an RTX 2060 (6 GB VRAM, ~3.5 GB free with a desktop
session running). If training OOMs in the first minute, lower --batch to 4.

Usage:
    python scripts/train.py                       # defaults below
    python scripts/train.py --epochs 100 --batch 4
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "datasets" / "indoor-nav" / "data.yaml"


def main():
    ap = argparse.ArgumentParser(description="Fine-tune YOLO11n for Neural-Edge")
    ap.add_argument("--model", default="yolo11n.pt", help="starting weights (COCO-pretrained)")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    # 6 GB VRAM: batch 8 is the safe ceiling at 640 px with a desktop session open.
    ap.add_argument("--batch", type=int, default=8)
    # Windows + PyTorch dataloaders: high worker counts are a common hang source.
    ap.add_argument("--workers", type=int, default=4)
    # Stop early if val mAP has not improved for this many epochs -- saves GPU hours
    # and guards against overfitting on a 2.8k-image set.
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="indoor-nav-v1", help="run name under runs/detect/")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"data.yaml not found at {args.data}\n"
            "See docs/datasets.md for how to obtain and place the dataset."
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        name=args.name,
        device=args.device,
        # Deterministic-ish seed so a re-run lands in the same neighbourhood.
        seed=0,
        # val=True is the default; kept explicit because the val mAP it produces
        # is the accuracy number every later pruning/quantization step is judged against.
        val=True,
    )

    weights = REPO_ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print("\n=== training complete ===")
    print(f"  best weights : {weights}")
    print("  next         : python scripts/benchmark.py --model "
          f"{weights} --source clip.mp4 --label finetuned-fp32 --measure 500")
    print("  (that re-baseline is the denominator for the Checkpoint 1 speedup claim)")


if __name__ == "__main__":
    main()
