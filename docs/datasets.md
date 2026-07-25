# Datasets

`datasets/` is gitignored (too large for the repo). This file records provenance so any
result in `results/` can be reproduced from a clean clone.

## indoor-nav (primary — W2 onward)

| Field | Value |
|---|---|
| Source | [Roboflow Universe — akhash/indoor-navigation-xs4of, v10](https://universe.roboflow.com/akhash/indoor-navigation-xs4of/dataset/10) |
| License | CC BY 4.0 |
| Classes | 17 |
| Images | 2,844 train · 105 valid · 62 test |
| Format | YOLO (Ultralytics) |
| Local path | `datasets/indoor-nav/` |

**Classes:** accessibility, door, elevator, elevator sign, exit sign, fire alarm,
fire extinguisher, handle, left arrow, men-s washroom, person, push handle,
right arrow, stair sign, trash can, water dispenser, women-s washroom.

**Why this dataset:** indoor navigation aids and obstacles — a plausible payload for a
service robot operating in a building, which is the deployment story this project models.
Selection criteria were pragmatic (1k–5k images, 3–10+ classes, pre-split, permissive
licence), because the research contribution here is the optimization pipeline, not
detection accuracy on a novel domain.

**Known limitation:** the validation split is small (105 images), so mAP is noisier than
ideal — differences under roughly 1–2 mAP points between pruning levels should not be
treated as significant. Recorded here so the ablation table in `results/` is read with
that caveat in mind.

### Reproducing

1. Download v10 from the URL above, export format **YOLOv11**.
2. Unzip to `datasets/indoor-nav/` so that `train/`, `valid/`, `test/` sit alongside `data.yaml`.
3. Rewrite the `train`/`val`/`test` keys in `data.yaml` as absolute paths.

## Benchmark clip

`clip.mp4` (gitignored): 1920×1080, 50 fps, 623 frames (12.5 s) of an indoor corridor with
doors and people walking through.

| Field | Value |
|---|---|
| Source | [Pixabay — "Office People Business Work Team", video 39890](https://pixabay.com/videos/office-people-business-work-team-39890/) |
| License | Pixabay Content License (free to use, no attribution required) |

Used as the fixed input for every latency measurement so all variants see identical frames.
Chosen for domain match (indoor corridor, doors, signage) and because detection count varies
across the clip — the opening frames are an empty corridor, later frames contain people —
so NMS cost is exercised realistically rather than on uniformly easy input. Frames are pre-resized to 640×640 at load — see the
`load_frames` docstring in `scripts/benchmark.py` for why preprocessing is excluded from
the timed loop.
