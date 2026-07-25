# Neural-Edge - Devlog

One short entry per work session: what I did, why, what blocked me, what's next. This is raw material for the final report - write it rough, write it same-day.

---

## 2026-07-05 - Project planning
- Full portfolio plan created.
- Environment verified.
- Next (W1): install Ubuntu WSL distro, Docker GPU passthrough smoke test, PyTorch 60-min blitz, run pretrained yolo11n on webcam.

## 2026-07-19 - W1 environment cleared (late), baseline model running

**Done**
- Native Windows Python env; PyTorch installed with CUDA build.
  `torch.cuda.is_available()` --> True, device = NVIDIA GeForce RTX 2060.
- Ultralytics installed; pretrained `yolo11n` running live on webcam.

**Decisions**
- **Pulled the environment fallback: native Windows instead of WSL/Docker for training.**
  Rationale: >7 days past the plan's 2-day friction trigger; containers aren't needed
  until W4 (TensorRT) and W6 (compose deployment). Removes the GPU-passthrough layer
  entirely for W1-W3.
- Docker GPU passthrough is working (`nvidia-smi` inside a CUDA container).
  Parked as a known-good option for W4/W6.

**Observation**
- yolo11n labelled a pen as "toothbrush". Not a detection failure - COCO's 80-class
  vocabulary contains "toothbrush" but not "pen", so the nearest label wins. Concrete
  motivation for fine-tuning on a domain dataset (W2).

**Blocked**
- `scripts/benchmark.py` not started. Structure is clear (warm-up then discard, per-frame
  timing with `torch.cuda.synchronize()` before stopping the clock, p50/p95 + FPS).
  Gap is API familiarity, not concepts. Resuming tomorrow.

**Status vs plan**
- W1 exit criteria met at the end of W2 --> ~1 week behind. W2 (benchmark harness,
  FP32 baseline, fine-tune underway) targeted for Monday-Tueday night.

**Next**
- Assemble `benchmark.py`; record FP32 baseline; pick dataset; launch fine-tune overnight Mon.

## 2026-07-25 - W2 closed: benchmark harness, FP32 baseline, fine-tune launched

**Done**
- Finished `scripts/benchmark.py` from my earlier sketch. It decodes N frames once, runs
  100 warm-up inferences and throws them away, then times each of 500 frames individually
  with `torch.cuda.synchronize()` before stopping the clock. Results go to `results/` as JSON
  with the git commit, environment and config baked in, so any number I quote later can be
  traced back to the code that produced it.
- **First baseline recorded** (`results/fp32-baseline.json`) - pretrained yolo11n, FP32, 640 px:
  p50 **16.31 ms**, p95 19.76 ms, **60.0 FPS**, 2279 MB VRAM, 50.9 W mean draw on an RTX 2060.
- Dataset chosen and wired up: Roboflow "indoor-navigation" v10 - 2,844 train / 105 valid /
  62 test, 17 classes, CC BY 4.0. Provenance recorded in `docs/datasets.md` since `datasets/`
  is gitignored.
- Benchmark clip sourced from Pixabay: an indoor corridor with doors and signage, people
  walking through from the middle of the clip onward.
- Fine-tune launched: yolo11n, 60 epochs, batch 8, 640 px.

**Decisions**
- **Preprocessing pulled out of the timed loop.** The clip is 1080p and the model runs at
  640 px, so `predict()` was resizing inside the timer. That resize is a fixed CPU cost that
  doesn't change with precision, so it would appear identically in the FP32 and INT8 numbers
  and compress the speedup ratio I am trying to measure. Frames are now resized once at load,
  and each result file records `preprocessing_in_timer: false`. End-to-end pipeline latency
  is a separate measurement, planned for W6 where it actually means something.
- **Tonight's baseline is not the denominator for Checkpoint 1.** The pretrained model has an
  80-class COCO head; the fine-tuned one will have 17. A smaller head is slightly cheaper to
  run, so comparing pretrained-FP32 against fine-tuned-INT8 would credit quantization with a
  speedup that actually came from changing the architecture. The fine-tuned FP32 model gets
  its own baseline run and that is what every later claim divides by. Tonight's number stays
  as an out-of-the-box reference point, labelled as such.
- Deferred deliberately: 3-run repeats (the `--runs` flag exists; 3 runs become mandatory
  before any number is published) and mAP inside the harness (there is nothing fine-tuned to
  measure yet, and the `map50_95` key is already reserved in the schema so adding it later
  won't invalidate existing result files).

**Dataset audit**
- I was uneasy about dataset quality, so I wrote `scripts/audit_dataset.py` and checked before
  building anything on top of it. 8,935 boxes across 2,844 training images (3.1 per image),
  no missing label files, no unlabelled images. Class imbalance is 18.9x between the most and
  least common class, which is normal for a real dataset. The thinnest class is `elevator sign`
  at 60 boxes across 39 images; everything else clears 180. Box density is consistent across
  train/val/test (2.9-3.1 per image), so the validation split looks representative.
- Conclusion: keep this dataset. Expect weak per-class results on `elevator sign` specifically,
  and read that as a data limitation rather than a model or pruning failure when it shows up
  in the W3 table.

**Open questions**
- The validation split is only 105 images, so val mAP will be noisy. Differences under about
  1-2 points between pruning levels shouldn't be read as real.
- I can follow the training loop end to end, but I don't yet have an intuition for how the
  individual loss terms (box / classification / DFL) trade off against each other, or what
  their relative magnitudes should look like on a healthy run. Watching them over this
  fine-tune to build that.

**What I learned**
- Working through dataset selection made the transfer-learning workflow concrete in a way
  reading about it hadn't: how much labelled public data already exists, how a model
  pretrained on generic COCO classes gets specialised onto a narrow domain, and how much of
  the practical work is data curation and evaluation methodology rather than model code.

**Status vs plan**
- W2 complete at the end of W3 --> still roughly one week behind. W3 (pruning) targeted for
  Sunday in a single long session.

**Next**
- Re-run the benchmark on the fine-tuned weights --> `results/finetuned-fp32.json`, the real
  denominator for Checkpoint 1.
- Audit per-class image counts in the training set.
- W3: write the pruning notes first, then torch-pruning at 20 % and 40 % sparsity, with a
  benchmark run after each.
