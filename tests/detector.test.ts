import { describe, it, expect } from "vitest";
import { decodeOutput, iou, nms, computeLetterbox, toFrameBox, type RawBox } from "../lib/detector";

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
    const out = decodeOutput(data, 2, 3, 0.5);
    expect(out).toHaveLength(1);
    expect(out[0]).toEqual({ cls: 1, score: 0.75, x1: 75, y1: 80, x2: 125, y2: 120 });
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
