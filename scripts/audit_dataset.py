"""
Audit a YOLO-format dataset before trusting any accuracy number measured on it.

Reports, per split: image count, label count, images with no annotations, and the
per-class box distribution. The point is to know *before* the ablation table whether
a weak mAP on some class means "the model failed" or "there were 12 examples".

Usage:
    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --data datasets/indoor-nav/data.yaml
"""

import argparse
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "datasets" / "indoor-nav" / "data.yaml"


def audit_split(images_dir: Path, names: list[str]):
    labels_dir = images_dir.parent / "labels"
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    class_counts = Counter()
    images_per_class = Counter()   # an image counts once per class present in it
    empty_images = 0
    missing_labels = 0
    total_boxes = 0

    for img in image_files:
        label_path = labels_dir / f"{img.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue

        lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
        if not lines:
            empty_images += 1  # background image: valid, but worth knowing how many
            continue

        seen_here = set()
        for ln in lines:
            cls_id = int(float(ln.split()[0]))
            class_counts[cls_id] += 1
            seen_here.add(cls_id)
            total_boxes += 1
        for cls_id in seen_here:
            images_per_class[cls_id] += 1

    return {
        "images": len(image_files),
        "boxes": total_boxes,
        "empty_images": empty_images,
        "missing_labels": missing_labels,
        "class_counts": class_counts,
        "images_per_class": images_per_class,
    }


def main():
    ap = argparse.ArgumentParser(description="Audit a YOLO dataset's class balance")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text())
    names = cfg["names"]

    reports = {}
    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        images_dir = Path(cfg[split])
        if not images_dir.exists():
            print(f"[warn] {split}: path does not exist -> {images_dir}")
            continue
        reports[split] = audit_split(images_dir, names)

    for split, r in reports.items():
        print(f"\n=== {split} ===")
        print(f"  images: {r['images']}   boxes: {r['boxes']}   "
              f"avg boxes/image: {r['boxes'] / max(r['images'], 1):.1f}")
        if r["empty_images"]:
            print(f"  images with no objects: {r['empty_images']}")
        if r["missing_labels"]:
            print(f"  MISSING label files: {r['missing_labels']}")

    train = reports.get("train")
    if not train:
        return

    print(f"\n=== per-class (train) ===")
    print(f"{'class':<22} {'boxes':>7} {'images':>7}")
    print("-" * 38)
    rows = sorted(
        ((names[i], train["class_counts"].get(i, 0), train["images_per_class"].get(i, 0))
         for i in range(len(names))),
        key=lambda r: r[1],
    )
    for name, boxes, imgs in rows:
        flag = "  <-- thin" if boxes < 50 else ""
        print(f"{name:<22} {boxes:>7} {imgs:>7}{flag}")

    counts = [c for _, c, _ in rows]
    if counts and counts[-1] > 0:
        print(f"\n  imbalance ratio (most/least common): {counts[-1] / max(counts[0], 1):.1f}x")
        thin = [n for n, c, _ in rows if c < 50]
        if thin:
            print(f"  classes under 50 boxes: {', '.join(thin)}")
            print("  -> expect low per-class mAP on these; not necessarily a model failure.")


if __name__ == "__main__":
    main()
