/**
 * Purpose : Thin wrappers over the browser's built-in text-to-speech (speechSynthesis)
 *           and speech-to-text (SpeechRecognition). Client-only.
 * Depends : Web Speech API (Chrome Android + Safari iOS support it; Firefox mobile does not).
 * Question you should be able to answer: why do we need isSpeaking() for the microphone logic?
 */
export type Recognizer = { start(): void; stop(): void };

let speaking = false;

export function isSpeaking(): boolean { return speaking; }

export function stopSpeaking(): void {
  window.speechSynthesis.cancel();
  speaking = false;
}

export function speak(text: string, opts: { interrupt?: boolean } = {}): Promise<void> {
  return new Promise((resolve) => {
    if (opts.interrupt) window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.1;
    u.onstart = () => { speaking = true; };
    u.onend = () => { speaking = false; resolve(); };
    u.onerror = () => { speaking = false; resolve(); };
    window.speechSynthesis.speak(u);
  });
}

// Minimal typing for the prefixed SpeechRecognition constructor (not in TS's DOM lib on all versions).
type SRResultEvent = { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> };
type SRInstance = {
  continuous: boolean; interimResults: boolean; lang: string;
  onresult: ((e: SRResultEvent) => void) | null; onend: (() => void) | null; onerror: ((e: { error: string }) => void) | null;
  start(): void; stop(): void; abort(): void;
};
type SRCtor = new () => SRInstance;

function getCtor(): SRCtor | null {
  const w = window as unknown as { SpeechRecognition?: SRCtor; webkitSpeechRecognition?: SRCtor };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function createRecognizer(opts: {
  continuous: boolean;
  onResult: (text: string, isFinal: boolean) => void;
  onEnd: () => void;
  onError: (error: string) => void;
}): Recognizer | null {
  const Ctor = getCtor();
  if (!Ctor) return null;
  const r = new Ctor();
  r.continuous = opts.continuous;
  r.interimResults = true;
  r.lang = "en-IN";
  r.onresult = (e) => {
    let text = "";
    let isFinal = false;
    for (let i = e.resultIndex; i < e.results.length; i++) {
      text += e.results[i][0].transcript;
      if (e.results[i].isFinal) isFinal = true;
    }
    opts.onResult(text.trim(), isFinal);
  };
  r.onend = () => opts.onEnd();
  r.onerror = (e) => opts.onError(e.error);
  return { start: () => { try { r.start(); } catch { /* already started */ } }, stop: () => r.stop() };
}

let audioCtx: AudioContext | null = null;
/** Short 880 Hz beep: "I heard you, asking now." Created lazily — AudioContext needs a prior user gesture on iOS. */
export function beep(durationMs = 120): void {
  try {
    audioCtx ??= new AudioContext();
    const o = audioCtx.createOscillator(); const g = audioCtx.createGain();
    o.frequency.value = 880; g.gain.value = 0.2;
    o.connect(g).connect(audioCtx.destination);
    o.start(); o.stop(audioCtx.currentTime + durationMs / 1000);
  } catch { /* no audio available — the spoken answer still comes */ }
}
