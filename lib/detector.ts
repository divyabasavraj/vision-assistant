/**
 * Purpose : Run our fine-tuned YOLO11n ONNX model in the browser and return boxes in frame fractions.
 *           Pure parts (decodeOutput, iou, nms, letterbox maths) are tested in tests/detector.test.ts.
 * Depends : onnxruntime-web (loaded lazily on the client), public/models/best.onnx + labels.json.
 * Question you should be able to answer: what is letterboxing and why do we need to undo it?
 */
import type { Frame } from "./frameSource";

export type Box = { cls: number; label: string; score: number; cx: number; cy: number; w: number; h: number };
export type RawBox = { cls: number; score: number; x1: number; y1: number; x2: number; y2: number };
export type Letterbox = { scale: number; padX: number; padY: number; frameW: number; frameH: number };

/**
 * Global score bar. Measured, not guessed: mean F1 over the 904-image test split peaks here (0.661)
 * and falls to 0.642 at 0.5 - and F1 already flatters the higher bar, because it prices a miss and a
 * false alarm the same. For a walking aid a missed pole is worse than an imagined one.
 *
 * This was 0.5 for a while, to stop low-confidence guesses being narrated. The narrator now does that
 * job properly and for free: a spurious box flickers, so it never matches a track at IoU 0.4, never
 * survives three consecutive frames, and is never spoken. A real object at 0.38 sits still and gets
 * through. Filtering by confidence suppressed the noise by also suppressing the signal.
 *
 * See training/results/threshold_sweep.md.
 */
export const SCORE_THRESHOLD = 0.35;

/**
 * Classes whose measured curve wants a different operating point from the default.
 * - curb: the one the confusion matrix predicted wrong. Gives up 0.010 recall for 0.142 precision
 *   going 0.35 -> 0.50, the best trade in the table. Pole looks similar at a glance (+0.123
 *   precision) but costs 0.118 recall to get it, so pole stays on the default.
 * - person: the model's favourite guess for anything upright it does not know (32% of background
 *   false positives). Measured in the narrator's own bands, 0.40 keeps close-band recall at 0.816
 *   exactly, loses 0.022 near, and removes 19% of false persons; 0.50 would cost 0.057 close recall,
 *   the people right in front of you. See training/results/person_threshold.md.
 */
export const CLASS_THRESHOLD: Record<string, number> = { curb: 0.5, person: 0.4 };

/** Per-class bars in model class order, so decodeOutput can index by class id without a lookup. */
export function thresholdsFor(labels: string[]): number[] {
  return labels.map((l) => CLASS_THRESHOLD[l] ?? SCORE_THRESHOLD);
}
export const NMS_IOU = 0.45;
export const ORT_VERSION = "1.29.0"; // keep equal to the version in package.json (npm ls onnxruntime-web)

/** Model output is [4+nc, anchors]: rows 0-3 = cx,cy,w,h in input pixels, rows 4.. = one score per class. */
export function decodeOutput(data: Float32Array, numClasses: number, numAnchors: number, thresholds: number[]): RawBox[] {
  const out: RawBox[] = [];
  for (let a = 0; a < numAnchors; a++) {
    let cls = -1, score = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = data[(4 + c) * numAnchors + a];
      if (s > score) { score = s; cls = c; }
    }
    // The bar depends on which class won, so it cannot be applied until after the argmax.
    if (cls < 0 || score < (thresholds[cls] ?? SCORE_THRESHOLD)) continue;
    const cx = data[a], cy = data[numAnchors + a], w = data[2 * numAnchors + a], h = data[3 * numAnchors + a];
    out.push({ cls, score, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
  }
  return out;
}

export function iou(a: RawBox, b: RawBox): number {
  const iw = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const ih = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = iw * ih;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union <= 0 ? 0 : inter / union;
}

/** Non-maximum suppression: highest score first; drop any box overlapping a kept box of the same class. */
export function nms(boxes: RawBox[], iouThreshold: number): RawBox[] {
  const sorted = [...boxes].sort((p, q) => q.score - p.score);
  const kept: RawBox[] = [];
  for (const b of sorted) {
    if (kept.every((k) => k.cls !== b.cls || iou(k, b) <= iouThreshold)) kept.push(b);
  }
  return kept;
}

/** Shrink the frame to fit inside inputSize×inputSize and centre it; the rest is grey padding. */
export function computeLetterbox(frameW: number, frameH: number, inputSize: number): Letterbox {
  const scale = Math.min(inputSize / frameW, inputSize / frameH);
  return { scale, padX: (inputSize - frameW * scale) / 2, padY: (inputSize - frameH * scale) / 2, frameW, frameH };
}

/** Undo the letterbox: input pixels -> fractions (0-1) of the original camera frame, clamped to the frame. */
export function toFrameBox(r: RawBox, lb: Letterbox, labels: string[]): Box {
  const x1 = (r.x1 - lb.padX) / lb.scale / lb.frameW, x2 = (r.x2 - lb.padX) / lb.scale / lb.frameW;
  const y1 = (r.y1 - lb.padY) / lb.scale / lb.frameH, y2 = (r.y2 - lb.padY) / lb.scale / lb.frameH;
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  const cx1 = clamp(x1), cx2 = clamp(x2), cy1 = clamp(y1), cy2 = clamp(y2);
  return { cls: r.cls, label: labels[r.cls] ?? String(r.cls), score: r.score, cx: (cx1 + cx2) / 2, cy: (cy1 + cy2) / 2, w: cx2 - cx1, h: cy2 - cy1 };
}

// ---- browser-only part below ----
type Ort = typeof import("onnxruntime-web");
type Session = import("onnxruntime-web").InferenceSession;

export class Detector {
  backend = "unknown";
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private input: Float32Array;

  /** Per-class bars, resolved once at load rather than per frame. */
  public readonly thresholds: number[];

  private constructor(private ort: Ort, private session: Session, public labels: string[], public inputSize: number) {
    this.thresholds = thresholdsFor(labels);
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = inputSize;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
    this.input = new Float32Array(3 * inputSize * inputSize);
  }

  static async load(modelUrl: string, labelsUrl: string, inputSize = 640): Promise<Detector> {
    const ort = await import("onnxruntime-web");
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    const labels: string[] = await (await fetch(labelsUrl)).json();
    let session: Session, backend: string;
    try {
      session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["webgpu"] });
      backend = "webgpu";
    } catch {
      session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
      backend = "wasm";
    }
    const d = new Detector(ort, session, labels, inputSize);
    d.backend = backend;
    return d;
  }

  /** Draw the frame letterboxed onto the 640×640 canvas and unpack RGBA bytes into 3 float planes (0-1). */
  private preprocess(frame: Frame): Letterbox {
    const lb = computeLetterbox(frame.width, frame.height, this.inputSize);
    const S = this.inputSize;
    this.ctx.fillStyle = "#727272";
    this.ctx.fillRect(0, 0, S, S);
    this.ctx.drawImage(frame.image, lb.padX, lb.padY, frame.width * lb.scale, frame.height * lb.scale);
    const px = this.ctx.getImageData(0, 0, S, S).data;
    const plane = S * S;
    for (let i = 0; i < plane; i++) {
      this.input[i] = px[i * 4] / 255;
      this.input[plane + i] = px[i * 4 + 1] / 255;
      this.input[2 * plane + i] = px[i * 4 + 2] / 255;
    }
    return lb;
  }

  async detect(frame: Frame): Promise<Box[]> {
    const lb = this.preprocess(frame);
    const S = this.inputSize;
    const tensor = new this.ort.Tensor("float32", this.input, [1, 3, S, S]);
    const out = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const t = out[this.session.outputNames[0]];
    const [, rows, anchors] = t.dims;           // [1, 4+nc, 8400]
    const raw = decodeOutput(t.data as Float32Array, rows - 4, anchors, this.thresholds);
    return nms(raw, NMS_IOU).map((r) => toFrameBox(r, lb, this.labels));
  }
}
