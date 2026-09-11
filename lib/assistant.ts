/**
 * Purpose : The conductor. Owns the camera, the mouth (TTS) and the ears (trigger-word listener),
 *           and later the detector, narrator and Q&A. React only renders what this class reports via onState.
 * Depends : lib/frameSource.ts, lib/speech.ts, lib/listener.ts, lib/voiceCommands.ts.
 * Question you should be able to answer: why keep this logic out of the React component?
 */
import { CameraFrameSource, type FrameSource } from "./frameSource";
import { speak, isSpeaking, stopSpeaking } from "./speech";
import { startWakeWordListener } from "./listener";
import type { Command } from "./voiceCommands";

export type AssistantState = {
  status: "idle" | "starting" | "running" | "error";
  caption: string;        // last thing the app said
  heard: string;          // live transcript
  listening: boolean;     // trigger-word listener active
  narrationPaused: boolean;
  error?: string;
  backend?: string;       // "webgpu" | "wasm" (Task 6)
  fps: number;            // detector frames per second (Task 6)
  boxes: unknown[];       // detections (typed as Box[] in Task 6)
  sttRestarts: number;
};

export class Assistant {
  state: AssistantState = { status: "idle", caption: "", heard: "", listening: false, narrationPaused: false, fps: 0, boxes: [], sttRestarts: 0 };
  readonly frames: FrameSource;
  private stopListening: (() => void) | null = null;

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
    this.stopListening = startWakeWordListener({
      onCommand: (c) => void this.handleCommand(c),
      onTranscript: (t) => this.set({ heard: t }),
      shouldIgnore: () => isSpeaking(),
      onRestart: (n) => this.set({ sttRestarts: n }),
      onUnsupported: () => this.set({ listening: false, caption: "Voice not supported here — use the button" }),
    });
    this.set({ status: "running", listening: this.stopListening !== null });
    await this.say("camera ready");
  }

  stop(): void {
    this.stopListening?.();
    this.frames.stop();
    stopSpeaking();
    this.set({ status: "idle", listening: false });
  }

  async say(text: string, interrupt = false): Promise<void> {
    this.set({ caption: text });
    await speak(text, { interrupt });
  }

  pauseNarration(): void { this.set({ narrationPaused: true }); void this.say("paused", true); }
  resumeNarration(): void { this.set({ narrationPaused: false }); void this.say("listening", true); }

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
