"""
Export the fine-tuned detector to a TensorRT engine.

Pipeline: PyTorch .pt -> ONNX -> TensorRT .engine

What each stage is for:
  - ONNX is a framework-neutral description of the network graph. It is the handoff
    format: PyTorch writes it, TensorRT reads it.
  - TensorRT then *compiles* that graph for this specific GPU. It fuses layers
    (conv + batchnorm + activation collapse into one kernel), picks the fastest
    available kernel for each operation by actually timing candidates on the device,
    and lays out memory to suit the architecture. The result is a serialised
    ".engine" file that is hardware-specific: an engine built on this RTX 2060 will
    not load on a different GPU generation.

Precision modes:
  - fp16 : half-precision floats. Roughly free on Turing tensor cores, usually near
           lossless in accuracy. The sensible default.
  - int8 : 8-bit integers. Faster and smaller still, but the network has to be
           *calibrated* first -- TensorRT runs a few hundred representative images
           through it to learn the numeric range of each tensor, so it can map floats
           onto 256 integer levels without clipping the values that matter.

Usage:
    python scripts/export.py --precision fp16
    python scripts/export.py --precision int8 --calib-images 500
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "indoor-nav-v1" / "weights" / "best.pt"
DEFAULT_DATA = REPO_ROOT / "datasets" / "indoor-nav" / "data.yaml"


def main():
    ap = argparse.ArgumentParser(description="Export YOLO11n to a TensorRT engine")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="dataset yaml; INT8 draws its calibration images from here")
    ap.add_argument("--workspace", type=int, default=2,
                    help="GB of scratch space the builder may use while searching kernels")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    # imgsz is part of the filename: engines at different input sizes are different
    # artifacts, and omitting it means a 512 px build silently overwrites the 640 px one.
    out_path = out_dir / f"yolo11n-indoor-{args.precision}-{args.imgsz}.engine"

    model = YOLO(args.weights)

    print(f"[info] exporting -> TensorRT {args.precision.upper()} (imgsz={args.imgsz})")
    print("[info] the builder times candidate kernels on this GPU; expect a few minutes")

    kwargs = dict(
        format="engine",
        imgsz=args.imgsz,
        workspace=args.workspace,
        verbose=False,
    )
    if args.precision == "fp16":
        kwargs["half"] = True
    else:
        # int8=True makes Ultralytics run calibration using images drawn from `data`.
        kwargs["int8"] = True
        kwargs["data"] = args.data

    produced = Path(model.export(**kwargs))

    # Ultralytics writes the engine next to the weights; move it under models/ so all
    # deployable artefacts live in one place.
    shutil.move(str(produced), str(out_path))

    size_mb = out_path.stat().st_size / 1024**2
    print(f"\n=== {out_path.name} ===")
    print(f"  size : {size_mb:.1f} MB")
    print(f"  path : {out_path.relative_to(REPO_ROOT)}")
    print("\n  next: python scripts/benchmark.py --model "
          f"{out_path.relative_to(REPO_ROOT)} --source clip.mp4 "
          f"--label {args.precision}-trt --measure 500")


if __name__ == "__main__":
    main()
