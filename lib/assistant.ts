/**
 * Purpose : The conductor. Owns the camera, the mouth (TTS) and the ears (trigger-word listener),
 *           and later the detector, narrator and Q&A. React only renders what this class reports via onState.
 * Depends : lib/frameSource.ts, lib/speech.ts, lib/listener.ts, lib/voiceCommands.ts, lib/detector.ts, lib/narrator.ts.
 * Question you should be able to answer: why keep this logic out of the React component?
 */
import { CameraFrameSource, type FrameSource } from "./frameSource";
import { speak, isSpeaking, stopSpeaking } from "./speech";
import { startWakeWordListener } from "./listener";
import type { Command } from "./voiceCommands";
import { Detector, type Box } from "./detector";
import { createNarrator } from "./narrator";

export type AssistantState = {
  status: "idle" | "starting" | "running" | "error";
  caption: string;        // last thing the app said
  heard: string;          // live transcript
  listening: boolean;     // trigger-word listener active
  narrationPaused: boolean;
  error?: string;
  backend?: string;       // "webgpu" | "wasm"
  fps: number;            // detector frames per second (average of the last 20)
  boxes: Box[];           // latest detections, coordinates as fractions of the camera frame
  sttRestarts: number;
};

export class Assistant {
  state: AssistantState = { status: "idle", caption: "", heard: "", listening: false, narrationPaused: false, fps: 0, boxes: [], sttRestarts: 0 };
  readonly frames: FrameSource;
  private stopListening: (() => void) | null = null;
  private detector: Detector | null = null;
  private loopTimer: number | null = null;
  private frameTimes: number[] = [];
  private narrator = createNarrator();
  private answering = false;   // true while a question is being answered (Task 8) — narration yields

  constructor(private onState: (s: AssistantState) => void, frames?: FrameSource) {
    this.frames = frames ?? new CameraFrameSource();
  }

  private set(patch: Partial<AssistantState>) {
    this.state = { ...this.state, ...patch };
    this.onState(this.state);
  }

  async start(): Promise<void> {
    this.set({ status: "starting" });
    try {
      await this.frames.start();
    } catch (e) {
      this.set({ status: "error", error: "Camera permission denied or no camera: " + String(e) });
      return;
    }
    try {
      this.detector = await Detector.load("/models/best.onnx", "/models/labels.json");
      this.set({ backend: this.detector.backend });
    } catch (e) {
      this.set({ status: "error", error: "Model failed to load: " + String(e) });
      return;
    }
    this.stopListening = startWakeWordListener({
      onCommand: (c) => void this.handleCommand(c),
      onTranscript: (t) => this.set({ heard: t }),
      shouldIgnore: () => isSpeaking(),
      onRestart: (n) => this.set({ sttRestarts: n }),
      onUnsupported: () => this.set({ listening: false, caption: "Voice not supported here — use the button" }),
    });
    this.set({ status: "running", listening: this.stopListening !== null });
    this.scheduleLoop();
    await this.say("camera ready");
  }

  stop(): void {
    if (this.loopTimer !== null) clearTimeout(this.loopTimer);
    this.loopTimer = null;
    this.stopListening?.();
    this.frames.stop();
    stopSpeaking();
    this.set({ status: "idle", listening: false });
  }

  // Detection loop: one frame at a time, back-to-back (no fixed interval — the model sets the pace).
  private scheduleLoop() { this.loopTimer = window.setTimeout(() => void this.tick(), 0); }

  private async tick(): Promise<void> {
    if (this.state.status !== "running" || !this.detector) return;
    const frame = this.frames.getFrame();
    if (frame) {
      const t0 = performance.now();
      const boxes = await this.detector.detect(frame);
      this.frameTimes.push(performance.now() - t0);
      if (this.frameTimes.length > 20) this.frameTimes.shift();
      const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
      this.set({ boxes, fps: Math.round(1000 / Math.max(avg, 1)) });
      this.onBoxes(boxes, performance.now());
    }
    if (this.state.status === "running") this.scheduleLoop();
  }

  // Narration: the narrator decides if these boxes are worth a phrase; we only decide whether we may speak now.
  protected onBoxes(boxes: Box[], nowMs: number): void {
    if (this.state.narrationPaused || this.answering) return;
    const phrase = this.narrator.update(boxes, nowMs);
    if (!phrase) return;
    if (isSpeaking() && !phrase.interrupt) return;   // don't pile up phrases while one is still being spoken
    void this.say(phrase.text, phrase.interrupt);
  }

  async say(text: string, interrupt = false): Promise<void> {
    this.set({ caption: text });
    await speak(text, { interrupt });
  }

  pauseNarration(): void { this.set({ narrationPaused: true }); this.narrator.reset(); void this.say("paused", true); }
  resumeNarration(): void { this.set({ narrationPaused: false }); this.narrator.reset(); void this.say("listening", true); }

  async handleCommand(cmd: Command): Promise<void> {
    switch (cmd.kind) {
      case "pause": return this.pauseNarration();
      case "resume": return this.resumeNarration();
      case "empty": return this.say("I didn't catch that", true);
      case "describe": return this.ask("Describe what is in front of me in two short sentences.");
      case "read": return this.ask("Read any text you can see, exactly as written.");
      case "ask": return this.ask(cmd.question);
    }
  }

  // Replaced by the real /api/ask call in Task 8. For now: echo, so we can test the voice path end to end.
  async ask(question: string): Promise<void> {
    await this.say("you asked: " + question, true);
  }
}
