# Task 9 — ablations

*An ablation changes exactly one thing and re-measures, so the write-up can say **why** the final
configuration is right rather than only **that** it works.*

**Status: done, 2026-09-30.** The Sep 22 overnight queue on the M5 (MPS) was too slow and was
replaced by two Kaggle T4 kernels (`training/kaggle/`), which finished in ~1 h. Same protocol, same
seed, same test split. Per-arm output: `training/results/ablation_runs/<name>_test.json`.

## Protocol

Every arm starts from COCO `yolo11n.pt` (except the model-size arm), trains **12 epochs, batch 16,
640 px, seed 0**, with `close_mosaic` at 20 % of the budget, and is evaluated on the **same
untouched 904-image test split** with the same command. Only the named factor differs.

Two honest caveats, stated up front:

- **12 epochs, not 50.** ~6 min/epoch on this machine; four 50-epoch arms is 20 h. The arms rank a
  factor against a control measured under an identical budget, which is what an ablation is for.
  Every absolute number here therefore sits below v1's 0.667, and none of these arms is the shipped
  model.
- **Batch 16, not v1's 32.** yolo11s at 640 px does not fit next to batch 32 in 16 GB of unified
  memory. Giving one arm a different batch size would confound it, so all arms dropped to 16.

Reasoning behind each pinned setting: `v2_decisions.md`.

**Optimizer, found afterwards.** All four arms left `optimizer: auto`, and at 12 epochs Ultralytics
chose AdamW (lr 0.000526) for every one of them, where v1's 50 epochs got SGD at 0.01. The arms are
still comparable with each other, since all used the same optimizer. It is one more reason their
absolute numbers are not comparable with v1's.

## Results

| run | what differs | imgsz | params | mAP50 (test) | mAP50-95 | new-class mAP50 | ONNX MB | phone fps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline COCO yolo11n | not fine-tuned at all | 640 | 2.6 M | 0.264 | — | 0 | 10.6 | — |
| **n640_v1** (shipped) | 50 epochs, batch 32 | 640 | 2.6 M | **0.667** | 0.462 | — | 10.6 | 5 (iPhone 13, wasm) |
| **n640_v2** | 30 more epochs from v1 + oversampling | 640 | 2.6 M | 0.653 | 0.447 | 0.662 | 10.6 | — |
| n640_e12 *(control)* | — | 640 | 2.6 M | 0.634 | 0.429 | 0.609 | 10.6 | — |
| n320_e12 | **A: image size** | 320 | 2.6 M | 0.567 | 0.380 | 0.528 | 10.6 | — |
| s640_e12 | **B: model size** (yolo11s) | 640 | 9.4 M | **0.675** | 0.460 | 0.657 | 37.9 | not measured (CPU 47 ms vs 21) |
| n640_noaug_e12 | **C: augmentation off** | 640 | 2.6 M | 0.601 | 0.389 | 0.629 | 10.6 | — |
| n640_v1 int8 | **D: quantisation** | 640 | 2.6 M | — | — | — | **3.0** | — |

v1's new-class mAP50 is 0.670. Batch 16 on Kaggle too, so the arms stay comparable with each
other (the T4 would have fit 32).

"New-class mAP50" = mean over the 7 classes COCO does not have (auto, stairs_up, stairs_down, curb,
pole, door, crosswalk) — the classes that only exist because of the fine-tune.

## Ablation D — int8 quantisation (done, 2026-09-22)

Post-training *dynamic* quantisation of the exported ONNX: weights stored as 8-bit unsigned
integers instead of 32-bit floats, activations quantised on the fly at run time. No retraining, no
calibration data.

| | fp32 | int8 |
|---|---:|---:|
| file size | 10.6 MB | **3.0 MB** (3.5× smaller) |
| CPU latency, 640×640, 2 threads | 21 ms | **27 ms** |

**The size win is real; the speed win is not.** int8 came out ~30 % *slower*. Dynamic quantisation
has to quantise and dequantise activations at every layer boundary at run time, and on Apple silicon
the fp32 convolution kernels ONNX Runtime falls back to are already very well optimised — so the
integer path pays the conversion overhead without getting a fast integer kernel in return. The
lesson is that "int8 is faster" is a claim about a particular kernel library on particular hardware,
not a property of quantisation.

Measured with `onnxruntime` on the CPU provider while a training job was running, so treat the
absolute milliseconds as provisional; the *ratio* was measured under identical conditions.
For the browser it is the phone number that decides (Step 5), because ONNX Runtime **Web** uses a
different backend (WASM) from the desktop CPU provider.

Reproduce: `training/.venv/bin/python training/scripts/quantize.py public/models/best.onnx public/models/best_int8.onnx`

## Conclusion

**The shipped configuration, `n640_v1` (YOLO11n, 640 px, default augmentation, fp32), survives every
arm.**

- **A: 320 px costs 6.7 points** (and 8 on the new classes). Small, thin objects vanish at low
  resolution. Keep 640.
- **B: yolo11s buys 4.1 points** and beats v1 even at 12 epochs, but it is 3.6× the size, 2.3×
  slower on CPU (~2 fps expected on the phone), and at the app's thresholds its recall is *lower*
  than v1's (0.565 vs 0.581, close band 0.762 vs 0.773). Keep nano.
- **C: augmentation is worth 3.3 points.** With it off, the new-class mean is slightly higher
  (0.629 vs 0.609), possibly because mosaic distorts thin/flat classes. Noted, not acted on.
- **D: int8 is smaller but slower** on this hardware. Keep fp32.
- **v2 (0.653) did not beat v1**: `optimizer: auto` switched SGD → AdamW and re-ran warm-up, which cost more than 30 epochs won back,
  and oversampling traded common-class recall for rare-class precision. Full analysis:
  `docs/PROJECT-LOG.md` §8.8.

## Other detector families (2026-10-06)

Same protocol (12 epochs, batch 16, 640 px, seed 0, AdamW 0.000526 via `auto`), Kaggle T4, job
`push.sh arch`. Question: are the weak classes (pole, curb, bicycle, crosswalk) a data or a model
problem?

| run | family | params | GFLOPs | mAP50 | mAP50-95 | new-class mAP50 | T4 inference | pole | curb | bicycle | crosswalk |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n640_e12 *(control)* | YOLO11n | 2.6 M | 6.5 | 0.634 | 0.429 | 0.609 | 3.8 ms | 0.409 | 0.465 | 0.515 | 0.461 |
| y26n640_e12 | YOLO26n | 2.5 M | 5.9 | 0.646 | 0.443 | 0.606 | 3.8 ms | 0.431 | 0.478 | 0.514 | 0.413 |
| rtdetrl640_e12 | RT-DETR-l | 32.8 M | 110 | **0.738** | **0.526** | **0.730** | 38.2 ms | **0.588** | **0.587** | **0.613** | 0.466 |

- **YOLO26n ≈ control.** A newer head at the same size does not fix the weak classes.
- **RT-DETR-l +0.104**, mostly via recall. The same data supports much better pole, curb and
  bicycle, so for those three nano's capacity is the limit, not the data.
- **crosswalk is flat for every model**, so it is limited by data or labels.
- Not shipped: 17× nano's compute, under 0.5 fps on the phone (estimated). Future work: distil it
  into nano. Full analysis: `docs/PROJECT-LOG.md` §8.9b.
