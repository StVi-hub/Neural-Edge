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

## 2026-07-26 - W3: fine-tune evaluated, structured pruning descoped after root-cause analysis

**Fine-tune results (60 epochs, ~50 min)**
- mAP50-95 **0.566**, mAP50 0.751, precision 0.886, recall 0.670.
- Worth being precise about what this means: the pretrained COCO model scores effectively
  zero on this dataset, because COCO has no "exit sign", "fire extinguisher" or "push handle"
  classes at all. So the comparison isn't 0.566 against some smaller number - it's 0.566
  against a model that structurally cannot represent the task.
- Recall (0.670) trails precision (0.886) by a wide margin: the model misses objects more
  often than it invents them. Consistent with the class imbalance found in yesterday's audit.

**Re-baseline (`results/finetuned-fp32.json`)**
- Fine-tuned FP32: p50 **15.851 ms**, p95 19.65 ms, **61.2 FPS**, 2256 MB, 45.6 W.
- Against yesterday's pretrained baseline (16.31 ms / 60.0 FPS) that is 2.8 % faster - and
  the cause is the 17-class head replacing the 80-class one, not any optimisation work.
  Confirms the decision to re-baseline: keeping the old denominator would have folded 2.8 %
  of free architectural speedup into the Checkpoint 1 claim.

**Structured pruning: attempted, root-caused, descoped**

Ran the plan's W3 pruning task. It failed silently at first - the script completed, printed a
successful forward pass, saved a model, and reported 0.0 % of parameters removed, with no
error raised. Debugging path:

1. First hypothesis: the ignore list was too coarse. I had protected the whole `Detect`
   module, and YOLO's neck concatenates backbone features, so a "these channels are fixed"
   constraint can propagate backwards across concat groups and freeze the network. Narrowed
   the protection to only the layers whose output shape is semantically fixed (`cv2[i][-1]`
   for the 4x reg_max box outputs, `cv3[i][-1]` for the 17 class scores, and `dfl.conv`).
   Still 0 %. Rejected.
2. Second hypothesis: eval vs train mode changes what gets traced. Tested as a 2x2 matrix
   ({eval, train} x {coarse, fine ignore list}) rather than one variable at a time. All four
   combinations returned 0 prunable groups. Both hypotheses eliminated in one pass.
3. Two dead hypotheses meant I was debugging at the wrong altitude - configuring the pruner
   when the problem was underneath it. Queried torch-pruning's `DependencyGraph` directly:
   **1 group total, and the model's first convolution was not in the graph at all.** That
   reframed the question from "why won't it prune?" to "why is the graph empty?".
4. **Root cause: Ultralytics loads checkpoints with `requires_grad=False` on all 256
   parameters.** torch-pruning discovers layer connectivity by walking the autograd graph,
   and autograd only records operations on tensors that require gradients. No gradients, no
   recorded graph, nothing to prune. An inference-side optimisation silently disabling a
   tool that depends on training machinery.
5. Forcing `requires_grad_(True)` fixed that blocker - tracing then began, and died with a
   **`MemoryError`** inside `torch_pruning/dependency/index_mapping.py`. The dependency
   resolver allocates a Python object per channel index and re-maps those lists at every
   coupling; YOLO11's C3k2 blocks chain those couplings deeply enough that memory use
   explodes. It exhausted **18 GB of available RAM** to analyse a 2.6M-parameter model, on
   both CPU and GPU. That is a library scalability limit on this architecture, not a
   hardware constraint - no plausible machine fixes it.

**Decision: descoped to winter, per the rule written into the roadmap in July.**
- Checkpoint 1 (>= 3x speedup) is reachable through INT8 TensorRT alone; pruning was always
  the optional second multiplier, which is exactly why the descope rule existed in advance.
- Deciding this by executing a pre-committed rule, rather than by how invested I felt after
  two hours of debugging, is the point of having written the rule down before starting.
- Even a success would not have fit: the full W3 cycle needs four more traces plus recovery
  fine-tunes. At the observed cost per trace, the technique was unusable within the session
  regardless of whether it eventually completed.
- Winter options, in rough order of promise: pin an older torch-pruning / Ultralytics pair,
  try YOLOv8 (much better tested with this library) instead of YOLO11, or write the
  dependency handling manually for the specific blocks involved.

**What I learned**
- Structured pruning removes whole channels (filters), so the model is physically smaller and
  genuinely faster on ordinary hardware. Unstructured pruning zeroes individual weights: the
  tensor keeps its shape, and without sparse-kernel support nothing gets faster. That
  distinction is the reason this project targets channels.
- Pruning is always prune -> accuracy drops -> short recovery fine-tune -> most of it returns.
  The surviving filters have to redistribute work the removed ones were doing.
- Silent failures are worse than crashes. A 0.0 % result with no exception cost far more time
  than an error message would have. Checking the *magnitude* of a result, not just its
  absence of errors, is the lesson.
- When two hypotheses in a row are wrong, the problem is probably a layer below where I'm
  looking.

**Open questions carried forward**
- `docs/learning-notes/pruning.md` is still owed, and writing it is the point: setting out
  structured vs unstructured pruning and the prune -> recover cycle in my own words is the
  test of whether I actually hold the concepts or merely recognise them.
- The deeper gap is architectural. I do not yet have a mental model of *why* YOLO11's C3k2
  blocks couple tightly enough that dependency resolution explodes, or of how torch-pruning
  represents those coupled groups internally. That is what to study before retrying in
  winter - without it, any fix would be guesswork rather than reasoning.

**Status vs plan**
- W2 fully closed. W3 partially closed: pruning attempted and documented, ablation table not
  produced. Moving to W4 (INT8 + TensorRT) with the unpruned fine-tuned model as input, which
  the roadmap already sanctioned as the fallback path.

**Next**
- W4: export ONNX -> TensorRT FP16 engine -> benchmark; then INT8 with a calibration set from
  the training images -> benchmark.
- Build the comparison table: FP32 PyTorch / FP16 TRT / INT8 TRT, all against the 15.851 ms
  fine-tuned denominator.
- Wire mAP measurement into the benchmark harness - Checkpoint 1 is a speedup claim *at a
  stated accuracy cost*, and the accuracy half is still missing.

### Same day, later - W4 started: TensorRT installed, FP16 engine built and measured

**Done**
- Installed TensorRT 11.1 plus the ONNX toolchain; wrote `scripts/export.py` for the
  .pt -> ONNX -> .engine pipeline with an `fp16` / `int8` switch.
- Added mAP measurement to the benchmark harness behind a `--val-data` flag. It runs the
  Ultralytics validator in a separate function, deliberately outside the timing loop -
  validation streams the whole val split with NMS and metric bookkeeping, which has nothing
  to do with per-frame inference cost and would corrupt both numbers if mixed in.
- Built the FP16 engine (141 s of kernel search, 6.8 MB) and benchmarked it.

**FP16 result**
- p50 **8.01 ms**, **121.1 FPS**, 2683 MB VRAM, 48.1 W, mAP50-95 0.5604.
- Against the 15.851 ms fine-tuned FP32 denominator that is a **1.98x speedup** - so FP16
  alone delivers two thirds of the Checkpoint 1 target, and INT8 needs to contribute roughly
  another 1.5x.
- Note VRAM went *up* (2256 -> 2683 MB): the TensorRT engine allocates its own activation
  and workspace memory. Faster does not automatically mean smaller in every dimension, which
  is worth remembering for the W5 controller where VRAM headroom matters.
  **[RETRACTED 2026-07-30 - this was an artefact of an unlocked GPU clock. Measured
  properly, TensorRT engines use roughly HALF the VRAM of the PyTorch model. See the
  2026-07-30 entry.]**
- The accuracy comparison is **not yet valid**. 0.5604 was measured by our validator; the
  ~0.566 from training came from Ultralytics' internal validation loop. Comparing across two
  different measurement paths would not be a real delta. Re-measuring the FP32 model through
  our own harness first, so the FP16 accuracy cost is a like-for-like number.

**Incident: TensorRT install silently replaced CUDA PyTorch with a CPU build**
- After installing TensorRT, inference failed with `torch.cuda.is_available(): False`. Cause:
  `torch` had been upgraded from 2.5.1+cu121 to 2.13.0+cpu. TensorRT pulls in
  `nvidia-modelopt`, which requires a newer torch, and pip resolved that from default PyPI -
  where the package named `torch` is the CPU-only build. CUDA builds exist only on PyTorch's
  own index. The install reported success; nothing surfaced until runtime.
- Fix: `pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
  --index-url https://download.pytorch.org/whl/cu121`. Verified 2.5.1+cu121 with CUDA
  available again.
- Prevention, applied from here on: when installing ML tooling alongside a working CUDA
  stack, pin torch explicitly in the same command or install with `--no-deps` and add
  dependencies deliberately. The TensorRT engine itself was unaffected - engines are
  self-contained compiled binaries and do not call into PyTorch at inference time.

**Next**
- Re-measure fine-tuned FP32 through our harness with `--val-data` to get a like-for-like
  accuracy baseline.
- Export and benchmark INT8 (calibration pass over training images).
- Write `scripts/compare.py` to build the results table from the committed JSONs.
- Re-run the final configuration with `--runs 3` before publishing any headline number.

## 2026-07-30 - W4 closed: Checkpoint 1 met at ~3x, and three conclusions retracted

**Checkpoint 1: MET.** FP16 TensorRT at 512 px input runs at **4.87 ms p50 / 201 FPS**
against the fine-tuned FP32 baseline's 14.59 ms - **~3x** - at an accuracy cost of
mAP50-95 0.5764 -> 0.5492, a 4.7 % relative drop. Both halves of that sentence are the
claim; the speedup alone would be meaningless.

### Final results (GPU clock pinned at 1710 MHz, 3 runs each)

`nvidia-smi -lgc 1700,1700` sets the min and max graphics clock to the same value, pinning
it rather than defining a range. The GPU settled on 1710 MHz - clocks exist only in discrete
bins and 1700 is not one of them. Every run recorded 1710-1710 MHz, 0.0 % spread, which is
the confirmation that the lock actually took effect.

| variant | p50 ms | FPS | speedup | mAP50-95 | FPS/W |
|---|---|---|---|---|---|
| fp16-trt-512 | 4.87 | 201.1 | **3.00x** | 0.5492 (-0.0272) | 3.50 |
| fp16-trt (640) | 6.13 | 154.1 | 2.38x | 0.5600 (-0.0164) | 2.63 |
| int8-trt (640) | 7.56 | 130.0 | 1.93x | 0.5537 (-0.0227) | 2.12 |
| finetuned-fp32 | 14.59 | 65.3 | 1.00x | 0.5764 | 1.06 |

Efficiency scales more strongly than latency: 3.3x better FPS/Watt at 512 px. For a
project whose thesis is energy-constrained inference, that is arguably the headline
number rather than the speedup.

### The methodological failure that dominated this session

Every measurement before tonight was taken with the GPU clock unlocked. The RTX 2060
idles at ~1110 MHz and boosts to 2100 MHz depending on temperature and power headroom,
which produced 8-15 % run-to-run variance. I never actually locked them, so from now on I will include it in my benchmarking procedure seeing the big impact.

**Three conclusions I had already written down turned out to be artefacts of that noise:**

1. *"INT8 is marginally faster than FP16."* Wrong. INT8 is 23 % slower (7.56 vs 6.13 ms).
   The single-run data had them within 2.5 % - inside the noise band.
2. *"TensorRT increases VRAM (2683 vs 2256 MB)."* Wrong, and backwards. Engines use
   roughly half the memory of the PyTorch model.
3. *"512 px input buys nothing, so the pipeline has hit an overhead floor and is no
   longer compute-bound."* Wrong. Unlocked, 512 px measured 1.3 % faster than 640 px;
   locked, it is **20 %** faster. The pipeline is still substantially compute-bound.

The third one mattered most, because it was a *diagnosis* rather than a data point. I had
concluded from it that further model-level optimisation was pointless and that only
overhead reduction could help - and I nearly acted on that. Decomposing the corrected
numbers (36 % fewer pixels giving 20 % less time) puts fixed overhead at ~2.6 ms and
compute at ~3.5 ms of the 6.13 ms at 640 px: overhead is significant but not dominant.

Lesson, stated plainly because it can cost a lot of time and resources: **an unlocked GPU clock does
not merely add uncertainty to a benchmark, it manufactures false findings that look
like results.** The variance was larger than several of the effects being measured, so
the noise was reliably mistaken for signal. Clock locking is now step one of the
protocol, and `benchmark.py` samples the SM clock every run, records min/max/spread in
the JSON, and warns above 3 % spread. A result with high spread is not publishable.

### INT8: measured, root-caused, rejected

INT8 is the slowest TensorRT variant at 1.93x - worse than FP16 on speed, accuracy *and*
power. Root cause, in two parts:

- Ultralytics' INT8 export bakes quantize/dequantize nodes into the ONNX graph.
  TensorRT 11.1's Myelin compiler cannot infer types across them ("Could not infer output
  types for operation: dequantize"), so it rejects the fused tactics and falls back to
  FP16 kernels - while still paying the Q/DQ overhead. Worst of both.
- I tried to bypass this by having TensorRT run its own entropy calibration from the
  clean FP32 graph. **That API no longer exists.** TensorRT 11 has no calibrator classes,
  no `BuilderFlag.INT8`, and no `BuilderFlag.FP16`; it offers
  `NetworkDefinitionCreationFlag.STRONGLY_TYPED` instead. TensorRT 11 is strongly-typed
  only - precision comes from the ONNX graph's dtypes and the builder obeys them. This
  completes the removal of implicit quantization, deprecated in TensorRT 10.

So Ultralytics was not misusing the API: baking Q/DQ into ONNX is now the *only* INT8
path. The failure is TensorRT 11.1 being unable to compile the Q/DQ pattern YOLO11's
architecture produces. Fixing it means either downgrading to TensorRT 10 (which still
has calibrators) or rewriting Q/DQ placement with nvidia-modelopt. Both are options to analyse.

**FP16 is the shipped engine.**

### Also done

- `scripts/build_engine.py`: builds engines through the TensorRT Python API directly,
  making each stage explicit (logger, builder, strongly-typed network, ONNX parser,
  config, serialize) rather than hidden behind Ultralytics or trtexec flags. `trtexec`
  is not available - pip ships the libraries and bindings but not the CLI tools.
- `scripts/compare.py`: generates the results table from committed JSONs, and flags
  single-run results and missing accuracy measurements rather than quietly tabulating
  them.
- `export.py` now includes `imgsz` in engine filenames; a 512 px build had silently
  overwritten the 640 px engine.
- 512 px is reported as a measured, accepted trade rather than a free win: 20 % faster
  for 4.7 % less accuracy. The 640 px engine remains the better default when accuracy
  matters; 512 px is what reaches 3x.

**Status vs plan**
- W4 closed on 30 Jul, inside its scheduled window (27 Jul - 2 Aug). Checkpoint 1 met.
  The W1-W2 slippage has been absorbed; the W3 descope is what bought the time back.
- W3 (pruning) remains descoped to winter.
- W5 (telemetry + controller) is next and is the piece that cannot be rushed.

**Next**
- W5: telemetry schema v1 (a public contract - Aether-Link implements the producing side
  in August), simulated and host providers, then the controller state machine with
  hysteresis and cooldown.
- Winter backlog gains: TensorRT 10 for a working INT8 calibrator path, and reducing
  per-call overhead (~2.6 ms of the 6.13 ms at 640 px) by calling the TensorRT runtime
  directly instead of through Ultralytics' per-frame Python path.
