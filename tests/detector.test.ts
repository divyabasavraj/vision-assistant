import { describe, it, expect } from "vitest";
import { decodeOutput, iou, nms, computeLetterbox, toFrameBox, thresholdsFor, SCORE_THRESHOLD, CLASS_THRESHOLD, type RawBox } from "../lib/detector";

// Build a fake model output [4+nc, anchors] laid out row-major like ONNX gives it.
function fakeOutput(nc: number, anchors: number, fill: (a: number) => number[] /* cx,cy,w,h,...scores */): Float32Array {
  const rows = 4 + nc;
  const data = new Float32Array(rows * anchors);
  for (let a = 0; a < anchors; a++) { const v = fill(a); for (let r = 0; r < rows; r++) data[r * anchors + a] = v[r]; }
  return data;
}

describe("decodeOutput", () => {
  it("keeps only anchors whose best class score passes the threshold", () => {
    // scores are powers of two so they survive Float32Array exactly and toEqual works
    const data = fakeOutput(2, 3, (a) => (a === 1 ? [100, 100, 50, 40, 0.125, 0.75] : [0, 0, 10, 10, 0.0625, 0.03125]));
    const out = decodeOutput(data, 2, 3, [0.5, 0.5]);
    expect(out).toHaveLength(1);
    expect(out[0]).toEqual({ cls: 1, score: 0.75, x1: 75, y1: 80, x2: 125, y2: 120 });
  });

  // A single global bar cannot serve fifteen classes whose precision/recall curves point in
  // opposite directions. See training/results/threshold_sweep.md.
  it("applies each class its own threshold", () => {
    // anchor 0 is class 0 at 0.375; anchor 1 is class 1 at 0.375. Same score, different bars.
    const data = fakeOutput(2, 2, (a) => (a === 0 ? [10, 10, 4, 4, 0.375, 0] : [20, 20, 4, 4, 0, 0.375]));
    const out = decodeOutput(data, 2, 2, [0.25, 0.5]);
    expect(out).toHaveLength(1);
    expect(out[0].cls).toBe(0);
  });
});

describe("per-class thresholds", () => {
  const LABELS = ["person", "dog", "car", "bicycle", "motorbike", "auto", "bus", "stairs_up", "stairs_down", "curb", "pole", "chair", "table", "door", "crosswalk"];

  it("defaults every class to the global threshold", () => {
    const t = thresholdsFor(["person", "pole", "crosswalk"]);
    expect(t).toEqual([SCORE_THRESHOLD, SCORE_THRESHOLD, SCORE_THRESHOLD]);
  });

  // Measured: curb gives up 0.010 recall for 0.142 precision going 0.35 -> 0.50, the best trade
  // in the table. Pole loses 0.118 recall for 0.123 precision, which is a bad deal for a walking
  // aid, so pole stays on the global default.
  it("raises curb above the default and leaves pole on it", () => {
    const t = thresholdsFor(LABELS);
    expect(t[LABELS.indexOf("curb")]).toBe(0.5);
    expect(t[LABELS.indexOf("pole")]).toBe(SCORE_THRESHOLD);
    expect(CLASS_THRESHOLD.curb).toBe(0.5);
  });

  it("uses 0.35 as the global default, not 0.5", () => {
    expect(SCORE_THRESHOLD).toBe(0.35);
  });

  it("falls back to the default for a label it has never heard of", () => {
    expect(thresholdsFor(["spaceship"])).toEqual([SCORE_THRESHOLD]);
  });
});

describe("iou / nms", () => {
  const a: RawBox = { cls: 0, score: 0.9, x1: 0, y1: 0, x2: 10, y2: 10 };
  const b: RawBox = { cls: 0, score: 0.8, x1: 5, y1: 0, x2: 15, y2: 10 };  // half overlap
  const c: RawBox = { cls: 1, score: 0.7, x1: 5, y1: 0, x2: 15, y2: 10 };  // same place, other class
  it("computes iou", () => { expect(iou(a, b)).toBeCloseTo(50 / 150, 5); expect(iou(a, { ...a, x1: 20, x2: 30 })).toBe(0); });
  it("drops overlapping lower-score boxes of the same class only", () => {
    expect(nms([b, a, c], 0.3)).toEqual([a, c]);
  });
  it("keeps both when overlap is under the threshold", () => { expect(nms([a, b], 0.5)).toEqual([a, b]); });
});

describe("letterbox mapping", () => {
  it("maps a box from 640-input space back to frame fractions", () => {
    const lb = computeLetterbox(1280, 720, 640);       // scale 0.5, padX 0, padY (640-360)/2 = 140
    expect(lb).toEqual({ scale: 0.5, padX: 0, padY: 140, frameW: 1280, frameH: 720 });
    const box = toFrameBox({ cls: 0, score: 0.9, x1: 160, y1: 140 + 90, x2: 480, y2: 140 + 270 }, lb, ["person"]);
    expect(box.label).toBe("person");
    expect(box.cx).toBeCloseTo(0.5); expect(box.cy).toBeCloseTo(0.5);
    expect(box.w).toBeCloseTo(0.5); expect(box.h).toBeCloseTo(0.5);
  });
});
