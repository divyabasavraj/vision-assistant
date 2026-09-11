/**
 * Purpose : Keep speech-to-text running all the time, look for the trigger word ("vision"),
 *           and hand the parsed Command to the assistant. Browsers stop recognition after a few
 *           seconds of silence, so we restart it automatically and count the restarts (for the report).
 * Depends : lib/speech.ts, lib/voiceCommands.ts.
 * Question you should be able to answer: how do we avoid the app answering its own spoken phrases?
 */
import { createRecognizer, type Recognizer } from "./speech";
import { parseCommand, type Command, TRIGGER_WORD } from "./voiceCommands";

export type ListenerOptions = {
  trigger?: string;
  onCommand: (cmd: Command) => void;
  onTranscript: (text: string) => void;   // for the on-screen caption
  shouldIgnore: () => boolean;            // true while the app itself is speaking
  onRestart?: (count: number) => void;
  onUnsupported?: () => void;
};

export function startWakeWordListener(opts: ListenerOptions): () => void {
  const trigger = opts.trigger ?? TRIGGER_WORD;
  let stopped = false;
  let restarts = 0;
  let rec: Recognizer | null = null;

  const boot = () => {
    rec = createRecognizer({
      continuous: true,
      onResult: (text, isFinal) => {
        if (opts.shouldIgnore()) return;
        opts.onTranscript(text);
        if (!isFinal) return;
        const cmd = parseCommand(text, trigger);
        if (cmd) opts.onCommand(cmd);
      },
      onEnd: () => {
        if (stopped) return;
        restarts += 1;
        opts.onRestart?.(restarts);
        setTimeout(() => { if (!stopped) rec?.start(); }, 300);
      },
      onError: (err) => {
        if (err === "not-allowed" || err === "service-not-allowed") { stopped = true; opts.onUnsupported?.(); }
        // "no-speech" and "aborted" are normal; onEnd will restart us.
      },
    });
    if (!rec) { opts.onUnsupported?.(); return; }
    rec.start();
  };

  boot();
  return () => { stopped = true; rec?.stop(); };
}
