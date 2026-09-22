# Task 9 — ablations

*An ablation changes exactly one thing and re-measures, so the write-up can say **why** the final
configuration is right rather than only **that** it works.*

**Status: runs in progress.** Started 2026-09-22 20:58 on the M5 (MPS), queued by
`training/scripts/run_queue.sh`. Numbers below marked — are not in yet; each arm writes
`training/runs/<name>_test.json` as it finishes.

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

## Results

| run | what differs | imgsz | params | mAP50 (test) | mAP50-95 | new-class mAP50 | ONNX MB | phone fps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline COCO yolo11n | not fine-tuned at all | 640 | 2.6 M | 0.264 | — | 0 | 10.6 | — |
| **n640_v1** (shipped) | 50 epochs, batch 32 | 640 | 2.6 M | **0.667** | 0.462 | — | 10.6 | 5 (iPhone 13, wasm) |
| **n640_v2** | 30 more epochs from v1 | 640 | 2.6 M | — | — | — | — | — |
| n640_e12 *(control)* | — | 640 | 2.6 M | — | — | — | — | — |
| n320_e12 | **A: image size** | 320 | 2.6 M | — | — | — | — | — |
| s640_e12 | **B: model size** (yolo11s) | 640 | 9.4 M | — | — | — | — | — |
| n640_noaug_e12 | **C: augmentation off** | 640 | 2.6 M | — | — | — | — | — |
| n640_v1 int8 | **D: quantisation** | 640 | 2.6 M | — | — | — | **3.0** | — |

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

*(written once the arms land — which configuration ships, and what each arm bought or cost)*
