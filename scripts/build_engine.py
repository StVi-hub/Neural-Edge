"""
Build a TensorRT engine directly with the Python API, bypassing Ultralytics' export.

Purpose: make every stage of the pipeline explicit, since both Ultralytics and trtexec
hide them behind a single call or a set of flags.

  1. Logger        - where TensorRT reports what it is doing
  2. Builder       - the compiler object
  3. Network       - an empty graph, populated by...
  4. OnnxParser    - ...reading the .onnx file into that graph
  5. BuilderConfig - workspace budget and build-time options
  6. Serialize     - compile to a .engine blob and write it to disk

FINDING (2026-07-30): TensorRT 11 removed calibrator-based INT8 quantization.

This script was originally written to test a hypothesis -- that Ultralytics' INT8 engine
was slow (7.34 ms vs FP16's 6.07 ms) because of *where* it placed quantize/dequantize
nodes in the ONNX graph, and that letting TensorRT run its own entropy calibration from
a clean FP32 graph would produce genuinely INT8 kernels instead of FP16 fallbacks.

The hypothesis turned out to be untestable on this stack. TensorRT 11.1 exposes:
  - no `IInt8EntropyCalibrator2` (or any calibrator class)
  - no `BuilderFlag.INT8`, and no `BuilderFlag.FP16` either
  - `NetworkDefinitionCreationFlag.STRONGLY_TYPED` instead

TensorRT 11 is strongly-typed only: precision is decided entirely by the data types in
the ONNX graph, and the builder obeys the graph rather than accepting precision requests.
This completes the removal of "implicit quantization", deprecated in TensorRT 10.

Consequences:
  - Ultralytics was not misusing the API. Baking Q/DQ nodes into ONNX is now the *only*
    supported INT8 path; their export was correct.
  - The Myelin "Could not infer output types for operation: dequantize" errors are
    TensorRT 11.1 failing to compile the Q/DQ pattern YOLO11's architecture produces.
  - Fixing INT8 therefore means either downgrading to TensorRT 10 (which still has
    calibrators) or rewriting Q/DQ placement with nvidia-modelopt. Both are winter work.
  - FP16 is the shipped engine: 6.07 ms, 2.37x, and better FPS/Watt than INT8 anyway.

Precision here comes from the ONNX you feed it. `best.fp16.onnx` yields an FP16 engine;
`best.onnx` yields FP32.

Usage:
    python scripts/build_engine.py --onnx runs/detect/indoor-nav-v1/weights/best.fp16.onnx
"""

import argparse
from pathlib import Path

import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = REPO_ROOT / "runs" / "detect" / "indoor-nav-v1" / "weights" / "best.fp16.onnx"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build(onnx_path: Path, out_path: Path, imgsz: int, workspace_gb: int):
    builder = trt.Builder(TRT_LOGGER)
    # STRONGLY_TYPED: the network's precisions come from the ONNX graph and the builder
    # honours them exactly, rather than being free to promote or demote layers. This is
    # the only mode TensorRT 11 offers -- see the module docstring.
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[info] parsing {onnx_path.name}")
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print(f"  parser error: {parser.get_error(i)}")
        raise SystemExit("ONNX parsing failed")

    print(f"[info] network: {network.num_layers} layers, "
          f"{network.num_inputs} input(s), {network.num_outputs} output(s)")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # No precision flags are set here, and none exist to set: TensorRT 11 has neither
    # BuilderFlag.FP16 nor BuilderFlag.INT8. Under STRONGLY_TYPED the graph's own dtypes
    # decide, so an FP16 ONNX produces an FP16 engine with nothing further to configure.

    print("[info] building engine "
          "(the builder times candidate kernels on this GPU; expect minutes)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed -- see TensorRT log above")

    out_path.write_bytes(serialized)
    print(f"\n=== {out_path.name} ===")
    print(f"  size : {out_path.stat().st_size / 1024**2:.1f} MB")
    print(f"  path : {out_path.relative_to(REPO_ROOT)}")
    print(f"\n  next: python scripts/benchmark.py --model {out_path.relative_to(REPO_ROOT)} "
          f"--source clip.mp4 --imgsz {imgsz} --label trt-manual "
          f"--measure 500 --runs 3 --val-data datasets/indoor-nav/data.yaml")


def main():
    ap = argparse.ArgumentParser(description="Build a TensorRT engine via the Python API")
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX),
                    help="precision comes from this graph's dtypes (STRONGLY_TYPED)")
    ap.add_argument("--imgsz", type=int, default=640, help="only used to label output")
    ap.add_argument("--workspace", type=int, default=2, help="GB")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"yolo11n-indoor-{args.imgsz}-manual.engine"

    build(Path(args.onnx), out_path, args.imgsz, args.workspace)


if __name__ == "__main__":
    main()
