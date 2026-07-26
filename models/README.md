# Models

Weights and engines are gitignored (binaries, and in the case of engines not portable
between machines at all). This manifest records what exists locally and how to rebuild it,
so every result in `results/` traces back to a reproducible artifact.

| File | Built from | Command | Notes |
|---|---|---|---|
| `runs/detect/indoor-nav-v1/weights/best.pt` | `yolo11n.pt` (COCO-pretrained) | `python scripts/train.py` | Fine-tuned on indoor-nav, 60 epochs, batch 8, 640 px. mAP50-95 0.566 at training time. |
| `models/yolo11n-indoor-fp16.engine` | `best.pt` | `python scripts/export.py --precision fp16` | TensorRT FP16. 6.8 MB, 141 s kernel search. |
| `models/pruned20.pt` | `best.pt` | `python scripts/prune.py --sparsity 0.20` | **Not a valid artifact.** 0 % of channels were actually removed — see the 2026-07-26 devlog entry. Kept only as a record of the attempt. |

## Why engines are not committed

A TensorRT engine is compiled for one specific GPU architecture and TensorRT version. The
builder times candidate kernels on the device it runs on and bakes the winners into the file.
An engine built here (RTX 2060, Turing, TRT 11.1) will not load on a different GPU generation
or a different TensorRT release. Committing one would ship a binary that only works on the
machine that produced it — hence rebuild-from-source rather than distribute.

## Environment note

Installing TensorRT can silently replace CUDA-enabled PyTorch with the CPU-only build from
PyPI (see the 2026-07-26 devlog entry). If `torch.cuda.is_available()` returns `False` after
any install, restore with:

```
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```
