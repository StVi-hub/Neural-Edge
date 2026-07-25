"""
Benchmark for Neural-Edge.

Measures inference latency (p50/p95), FPS, VRAM and GPU power so that every
variant (FP32 / pruned / FP16 / INT8) is compared under exactly the same rules.

Protocol (the single source of truth for every number this project claims):
  - Frames are decoded ONCE into RAM, so disk I/O never lands inside the timer.
  - WARMUP inferences are run and discarded (CUDA lazy init, autotuning,
    clock boost states all settle during this phase).
  - Each measured frame is timed individually, with torch.cuda.synchronize()
    before stopping the clock -- CUDA kernel launches are asynchronous, so
    without this we would be timing "how fast Python queued the work", not
    "how long the GPU took".
  - Results are written as JSON to results/ and committed.

Usage:
    python scripts/benchmark.py --label fp32-baseline
    python scripts/benchmark.py --model runs/detect/train/weights/best.pt --label finetuned-fp32
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# GPU telemetry (NVML). Optional: if pynvml is missing we still get latency.
# --------------------------------------------------------------------------
class GpuProbe:
    """Reads instantaneous VRAM use and power draw from the driver via NVML."""

    def __init__(self, index: int = 0):
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:  # pynvml missing, or no NVIDIA driver
            print(f"[warn] NVML unavailable ({exc}); power/VRAM will be null")

    def sample(self):
        """Return (vram_mb, power_w), or (None, None) if NVML is unavailable."""
        if self.handle is None:
            return None, None
        mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        power_mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
        return mem.used / 1024**2, power_mw / 1000.0

    def static_info(self):
        if self.handle is None:
            return {}
        name = self.pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(name, bytes):  # older pynvml returns bytes
            name = name.decode()
        return {
            "gpu_name": name,
            "driver_version": self.pynvml.nvmlSystemGetDriverVersion(),
        }


def load_frames(source: str, count: int, imgsz: int | None) -> list:
    """Decode up to `count` frames into memory once, before any timing starts.

    If `imgsz` is given, frames are resized here rather than inside predict().
    Rationale: the source clip is 1080p but the model runs at 640. That resize
    is a fixed CPU cost that does not change with precision, so leaving it in
    the timed loop would appear identically in the FP32 and INT8 numbers and
    compress the measured speedup ratio. We benchmark the model; end-to-end
    pipeline latency (with preprocessing) is measured separately in W6.
    """
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:  # short clip -> loop what we have rather than fail
            break
        if imgsz is not None:
            frame = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()

    if not frames:
        raise SystemExit(f"No frames decoded from: {source}")
    shape = frames[0].shape
    print(f"[info] loaded {len(frames)} frames from {source} at {shape[1]}x{shape[0]}")
    return frames


def git_commit() -> str | None:
    """Record which code produced this result -- reproducibility, not vanity."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def run_once(model, frames, imgsz, device, warmup, measure, probe):
    """One complete warm-up + measurement pass. Returns metrics for this run."""
    # --- warm-up: run and discard -------------------------------------------
    for i in range(warmup):
        model.predict(frames[i % len(frames)], imgsz=imgsz, device=device, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # --- measure ------------------------------------------------------------
    times_ms, power_samples, vram_samples = [], [], []
    for i in range(measure):
        frame = frames[i % len(frames)]

        t0 = time.perf_counter()
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # <-- without this we time the queue, not the GPU
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)

        if i % 10 == 0:  # sampling every frame would itself cost time
            vram, power = probe.sample()
            if power is not None:
                power_samples.append(power)
                vram_samples.append(vram)

    return {
        "latency_ms": {
            "p50": round(float(np.percentile(times_ms, 50)), 3),
            "p95": round(float(np.percentile(times_ms, 95)), 3),
            "mean": round(float(np.mean(times_ms)), 3),
            "stdev": round(float(statistics.stdev(times_ms)), 3) if len(times_ms) > 1 else 0.0,
        },
        # FPS from the MEAN latency, not from p50: throughput is about total
        # time for N frames, which is exactly what the mean encodes.
        "fps": round(1000.0 / float(np.mean(times_ms)), 2),
        "vram_mb": round(max(vram_samples), 1) if vram_samples else None,
        "power_w_mean": round(float(np.mean(power_samples)), 2) if power_samples else None,
        "frames_measured": measure,
    }


def main():
    ap = argparse.ArgumentParser(description="Neural-Edge inference benchmark")
    ap.add_argument("--model", default="yolo11n.pt", help="weights (.pt) or engine (.engine)")
    ap.add_argument("--source", default="0", help="video file path, or '0' for webcam")
    ap.add_argument("--label", required=True, help="short name for this variant, e.g. fp32-baseline")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--measure", type=int, default=1000)
    ap.add_argument("--runs", type=int, default=1, help="repeat count; use 3 for published numbers")
    ap.add_argument("--frames", type=int, default=200, help="distinct frames to hold in RAM")
    ap.add_argument("--notes", default="", help="free text stored in the JSON")
    ap.add_argument(
        "--end-to-end",
        action="store_true",
        help="keep resize inside the timed loop (pipeline latency instead of model latency)",
    )
    args = ap.parse_args()

    probe = GpuProbe()
    frames = load_frames(args.source, args.frames, None if args.end_to_end else args.imgsz)

    print(f"[info] loading model: {args.model}")
    model = YOLO(args.model)

    runs = []
    for r in range(args.runs):
        print(f"[info] run {r + 1}/{args.runs}: {args.warmup} warm-up + {args.measure} measured...")
        runs.append(run_once(model, frames, args.imgsz, args.device, args.warmup, args.measure, probe))
        print(f"       p50={runs[-1]['latency_ms']['p50']} ms  FPS={runs[-1]['fps']}")

    result = {
        "schema_version": SCHEMA_VERSION,
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": {
            "model": args.model,
            "source": args.source,
            "imgsz": args.imgsz,
            "device": args.device,
            "warmup": args.warmup,
            "measure": args.measure,
            "runs": args.runs,
            "precision": "fp32",  # updated by hand (or by flag) for fp16/int8 variants
            # False = frames pre-resized to imgsz, so the timer covers inference only.
            "preprocessing_in_timer": args.end_to_end,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **probe.static_info(),
        },
        # Accuracy is measured separately with model.val() until W4 wires it in.
        # The key exists now so the schema does not change when it does.
        "map50_95": None,
        "runs_detail": runs,
        "summary": {
            "p50_ms": round(statistics.median(r["latency_ms"]["p50"] for r in runs), 3),
            "p95_ms": round(statistics.median(r["latency_ms"]["p95"] for r in runs), 3),
            "fps": round(statistics.median(r["fps"] for r in runs), 2),
            "vram_mb": runs[0]["vram_mb"],
            "power_w_mean": runs[0]["power_w_mean"],
        },
        "notes": args.notes,
    }

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2))

    s = result["summary"]
    print(f"\n=== {args.label} ===")
    print(f"  p50 {s['p50_ms']} ms | p95 {s['p95_ms']} ms | {s['fps']} FPS")
    print(f"  VRAM {s['vram_mb']} MB | power {s['power_w_mean']} W")
    print(f"  -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
