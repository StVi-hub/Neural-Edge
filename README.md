# Neural-Edge — Adaptive Inference Under Power Constraints

> **MVP landing September 2026.** Active development.

Deep learning models get more accurate as they get larger. Robots get smaller, hotter, and
run on batteries. Neural-Edge is about closing that gap: taking an object detector and making
it fast enough, small enough, and *adaptive* enough to keep working when the hardware starts
running out of power.

The target scenario is field and disaster-response robotics — a machine that has to keep
seeing obstacles as its battery drains and its silicon throttles, degrading its own quality
gracefully instead of stalling.

## Part of a three-project system

| Project | Role |
|---|---|
| **Neural-Edge** (this repo) | Adaptive perception under power and thermal limits |
| [Aether-Link](https://github.com/StVi-Hub/Aether-Link) | Deterministic telemetry transport |
| [QKD-Bot](https://github.com/StVi-Hub/QKD-Bot) | Quantum-secured command channel |

Aether-Link delivers, over shared memory, the live telemetry that drives this project's
adaptive controller. The integration is a tracked milestone, not an aspiration.

## Platform

- **GPU:** NVIDIA RTX 2060 (Turing, 6 GB) — CUDA
- **Frameworks:** PyTorch, Ultralytics YOLO11, TensorRT
- **Deployment:** Docker

Power envelopes are emulated by **locking GPU clocks** (`nvidia-smi -lgc`) and measuring real
draw via NVML, since this card cannot be power-capped below roughly 125 W. Running the same
pipeline on Jetson hardware is planned future work, not a current claim.

## Roadmap to MVP

### Phase 1 — Compression
- [ ] Fine-tune YOLO11n on an obstacle-detection dataset
- [ ] Structured channel pruning (removes whole channels, so the tensors actually shrink and
      every GPU runs less work — unlike unstructured sparsity)
- [ ] INT8 post-training quantization, exported to a TensorRT engine
- [ ] **Target:** ≥3x latency/throughput improvement versus the FP32 PyTorch baseline —
      same GPU, fixed clocks, identical measurement protocol

### Phase 2 — Adaptive control
- [ ] Versioned telemetry schema (battery, CPU/GPU temperature, power budget)
- [ ] Simulated and host telemetry providers
- [ ] Controller as a finite state machine: Normal → Economy → Critical, with **hysteresis and
      cooldown** so it cannot oscillate around a threshold
- [ ] Actions: resolution reduction, frame skipping, engine swap — every transition logged
- [ ] **Target:** demonstrable graceful degradation — telemetry degrades mid-video, the
      pipeline visibly adapts and never stalls

### Phase 3 — Deployment and energy mapping
- [ ] Full pipeline containerized, one `docker compose up`
- [ ] Clock-lock sweep across emulated power envelopes
- [ ] **Target:** published FPS-per-Watt curves for FP32 / FP16 / INT8 / pruned variants

## Measurement protocol

Every number in this repository is produced by the same harness under the same conditions:
fixed GPU clocks, 100-frame warm-up discarded, 1000 measured frames, 3 runs. Reported as
p50 and p95 latency, FPS, mAP50-95, peak VRAM, and NVML power draw, serialized to JSON in
`results/`. A benchmark without a protocol is an anecdote.

## Current status

- Environment running: native Windows + CUDA, GPU confirmed available to PyTorch
- Pretrained YOLO11n running live inference
- Benchmark harness in progress

## Planned extensions (winter 2026–27)

Knowledge distillation from a larger teacher model, QAT versus PTQ comparison, deeper pruning
ablations, a YOLO11s comparison point, and re-running the full pipeline on Jetson Orin hardware.

---

*Development log: [`devlog.md`](devlog.md)*
