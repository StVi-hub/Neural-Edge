# Neural-Edge — Devlog

One short entry per work session: what I did, why, what blocked me, what's next. This is raw material for the final report — write it rough, write it same-day.

---

## 2026-07-05 — Project planning
- Full portfolio plan created.
- Environment verified.
- Next (W1): install Ubuntu WSL distro, Docker GPU passthrough smoke test, PyTorch 60-min blitz, run pretrained yolo11n on webcam.

## 2026-07-19 — W1 environment cleared (late), baseline model running

**Done**
- Native Windows Python env; PyTorch installed with CUDA build.
  `torch.cuda.is_available()` -> True, device = NVIDIA GeForce RTX 2060.
- Ultralytics installed; pretrained `yolo11n` running live on webcam.

**Decisions**
- **Pulled the environment fallback: native Windows instead of WSL/Docker for training.**
  Rationale: >7 days past the plan's 2-day friction trigger; containers aren't needed
  until W4 (TensorRT) and W6 (compose deployment). Removes the GPU-passthrough layer
  entirely for W1–W3.
- Docker GPU passthrough is working (`nvidia-smi` inside a CUDA container).
  Parked as a known-good option for W4/W6.

**Observation**
- yolo11n labelled a pen as "toothbrush". Not a detection failure — COCO's 80-class
  vocabulary contains "toothbrush" but not "pen", so the nearest label wins. Concrete
  motivation for fine-tuning on a domain dataset (W2).

**Blocked**
- `scripts/benchmark.py` not started. Structure is clear (warm-up then discard, per-frame
  timing with `torch.cuda.synchronize()` before stopping the clock, p50/p95 + FPS).
  Gap is API familiarity, not concepts. Resuming tomorrow.

**Status vs plan**
- W1 exit criteria met at the end of W2 -> ~1 week behind. W2 (benchmark harness,
  FP32 baseline, fine-tune underway) targeted for Monday–Tueday night.

**Next**
- Assemble `benchmark.py`; record FP32 baseline; pick dataset; launch fine-tune overnight Mon.
