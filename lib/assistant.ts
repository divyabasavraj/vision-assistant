/**
 * Purpose : The conductor. Owns the camera, the mouth (TTS) and the ears (trigger-word listener),
 *           and later the detector, narrator and Q&A. React only renders what this class reports via onState.
 * Depends : lib/frameSource.ts, lib/speech.ts, lib/listener.ts, lib/voiceCommands.ts, lib/detector.ts, lib/narrator.ts, lib/ask.ts.
 * Question you should be able to answer: why keep this logic out of the React component?
 */
import { CameraFrameSource, type FrameSource } from "./frameSource";
import { speak, isSpeaking, stopSpeaking, beep, cue } from "./speech";
import { startWakeWordListener } from "./listener";
import type { Command } from "./voiceCommands";
import { Detector, type Box } from "./detector";
import { createNarrator, type Zone } from "./narrator";
import { frameToJpeg, askServer } from "./ask";

/** Where each zone sits in the stereo field, for the cue tone that precedes every phrase. */
const ZONE_PAN: Record<Zone, number> = { left: -0.8, ahead: 0, right: 0.8 };

/**
 * "always" keeps speech-to-text running the whole session, so "vision, ..." works hands-free — but
 * Safari transcribes in the cloud, so every conversation nearby leaves the device. "hold" never starts
 * that listener: the mic only opens while the talk button is held. Convenience vs bystander privacy.
 */
export type MicMode = "always" | "hold";

export type AssistantState = {
  status: "idle" | "starting" | "running" | "error";
  micMode: MicMode;
  caption: string;        // last thing the app said
  heard: string;          // live transcript
  listening: boolean;     // trigger-word listener active
  narrationPaused: boolean;
  error?: string;
  backend?: string;       // "webgpu" | "wasm"
  fps: number;            // detector frames per second (average of the last 20)
  boxes: Box[];           // latest detections, coordinates as fractions of the camera frame
  sttRestarts: number;
  lastAskMs: number;      // round-trip of the last cloud question (0 = none yet)
};

export class Assistant {
  state: AssistantState = { status: "idle", micMode: "always", caption: "", heard: "", listening: false, narrationPaused: false, fps: 0, boxes: [], sttRestarts: 0, lastAskMs: 0 };
  readonly frames: FrameSource;
  private stopListening: (() => void) | null = null;
  private detector: Detector | null = null;
  private loopTimer: number | null = null;
  private frameTimes: number[] = [];
  private narrator = createNarrator();
  private answering = false;   // true while a question is being answered — narration yields

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
    if (this.state.micMode === "always") this.startListening();
    this.set({ status: "running" });
    this.scheduleLoop();
    await this.say("camera ready");
  }

  private startListening(): void {
    this.stopListening = startWakeWordListener({
      onCommand: (c) => void this.handleCommand(c),
      onTranscript: (t) => this.set({ heard: t }),
      shouldIgnore: () => isSpeaking(),
      onRestart: (n) => this.set({ sttRestarts: n }),
      onUnsupported: () => this.set({ listening: false, caption: "Voice not supported here — use the button" }),
    });
    this.set({ listening: this.stopListening !== null });
  }

  /** Switching to "hold" tears the listener down rather than ignoring it: the mic must actually close. */
  setMicMode(mode: MicMode): void {
    if (mode === this.state.micMode) return;
    this.set({ micMode: mode, heard: "" });
    if (mode === "hold") {
      this.stopListening?.();
      this.stopListening = null;
      this.set({ listening: false });
      void this.say("hold to talk", true);
    } else if (this.state.status === "running") {
      this.startListening();
      void this.say("always listening", true);
    }
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
    cue(ZONE_PAN[phrase.zone]);
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

  // One photo + the question go to /api/ask (cloud vision model). Narration pauses meanwhile and the
  // tracker is reset afterwards because the scene may have changed while we were away.
  async ask(question: string): Promise<void> {
    if (this.answering) return;
    this.answering = true;
    try {
      beep();
      stopSpeaking();
      const frame = this.frames.getFrame();
      if (!frame) { await this.say("camera not ready", true); return; }
      this.set({ caption: "asking…", heard: question });
      const result = await askServer(question, frameToJpeg(frame));
      this.set({ lastAskMs: result.ms });
      if ("answer" in result) await this.say(result.answer, true);
      else if (result.error === "offline") await this.say("no connection — questions need internet", true);
      else if (result.error === "retry") await this.say("please try again in a moment", true);
      else await this.say("something went wrong, try again", true);
    } finally {
      this.answering = false;
      this.narrator.reset();
    }
  }
}
