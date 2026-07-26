"""
Structured channel pruning for the fine-tuned YOLO11n detector.

Structured (not unstructured) pruning: whole output channels are physically removed
from conv layers, so the resulting model is genuinely smaller and genuinely faster on
ordinary hardware. Unstructured pruning zeroes individual weights, which shrinks the
model on paper but needs sparse-matrix kernel support to run any faster -- consumer
GPUs give you nothing for it. That distinction is the whole reason this project prunes
channels rather than weights.

Choices made here (see docs/learning-notes/pruning.md for the reasoning):
  - criterion    : L1 magnitude. Channels whose conv weights have the smallest L1 norm
                   are assumed to contribute least, and go first.
  - scope        : backbone + neck only. The detection head is left intact -- its DFL
                   and per-class outputs have shape constraints that break easily, and
                   the head is a small share of total FLOPs anyway.
  - distribution : global. All prunable channels are ranked network-wide and the least
                   important X% are removed, so some layers shrink far more than others.

Why a dependency graph is needed at all: removing an output channel from one conv
invalidates the matching input channel of every layer that consumes it, plus the
associated BatchNorm statistics, and residual/concat connections force whole groups of
layers to be pruned together. torch-pruning traces the model to work those groups out;
doing it by hand for a YOLO backbone is where this task goes wrong.

Usage:
    python scripts/prune.py --sparsity 0.20
    python scripts/prune.py --sparsity 0.40 --name pruned40
"""

import argparse
from pathlib import Path

import torch
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "indoor-nav-v1" / "weights" / "best.pt"


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser(description="Structured channel pruning for YOLO11n")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--sparsity", type=float, required=True, help="fraction of channels to remove, e.g. 0.20")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default=None, help="output name; defaults to pruned<pct>")
    args = ap.parse_args()

    name = args.name or f"pruned{int(args.sparsity * 100)}"
    out_dir = REPO_ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{name}.pt"

    # Load on CPU: tracing and surgery are simpler without device juggling, and the
    # recovery fine-tune will move it back to the GPU anyway.
    yolo = YOLO(args.weights)
    model = yolo.model
    model.eval()
    model.cpu()

    params_before = count_params(model)

    # The example input is what torch-pruning traces to build the dependency graph.
    example = torch.randn(1, 3, args.imgsz, args.imgsz)

    # Protect only the layers whose OUTPUT shape is semantically fixed:
    #   cv2[i][-1] -> 4 * reg_max box-regression outputs
    #   cv3[i][-1] -> nc class-score outputs
    #   dfl.conv   -> the fixed reg_max -> 1 projection
    #
    # Ignoring the whole Detect module instead is the obvious-looking move and it does
    # not work: YOLO's neck concatenates backbone and neck features, so the "cannot
    # change these channels" constraint propagates backwards across every concat group
    # and freezes the entire network (measured: 0 prunable groups). Protecting just the
    # final 1x1 convs leaves the head's interior and the whole backbone prunable, while
    # still guaranteeing the model emits 17 classes and 64 box bins.
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, Detect):
            for branch in list(m.cv2) + list(m.cv3):
                ignored_layers.append(branch[-1])
            if hasattr(m, "dfl"):
                ignored_layers.append(m.dfl.conv)
    print(f"[info] protecting {len(ignored_layers)} output layer(s) from pruning")

    # MagnitudeImportance(p=1) == L1 norm of each channel's weights.
    importance = tp.importance.GroupMagnitudeImportance(p=1)

    pruner = tp.pruner.MetaPruner(
        model,
        example,
        importance=importance,
        pruning_ratio=args.sparsity,
        global_pruning=True,        # rank channels network-wide, not per layer
        ignored_layers=ignored_layers,
        iterative_steps=1,          # single shot; recovery happens in the fine-tune
    )

    print(f"[info] pruning {args.sparsity:.0%} of channels (global, L1, head protected)...")
    pruner.step()

    params_after = count_params(model)
    reduction = 1.0 - params_after / params_before

    # Sanity check: the pruned graph must still produce correctly shaped outputs.
    with torch.no_grad():
        out = model(example)
    print(f"[info] forward pass OK after pruning (output type: {type(out).__name__})")

    # Ultralytics reloads via its own wrapper, so persist the whole YOLO object's model.
    yolo.model = model
    yolo.save(str(out_path))

    print(f"\n=== {name} ===")
    print(f"  params : {params_before:,} -> {params_after:,}  ({reduction:.1%} removed)")
    print(f"  saved  : {out_path.relative_to(REPO_ROOT)}")
    print("\n  next: recovery fine-tune, then benchmark. Pruned accuracy before recovery")
    print("        is expected to be poor -- that drop is what the fine-tune repairs.")


if __name__ == "__main__":
    main()
