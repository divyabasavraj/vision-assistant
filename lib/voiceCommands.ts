/**
 * Purpose : Turn what the microphone heard ("um vision, read the sign") into a Command.
 *           Pure string logic — no browser APIs — so it is unit-tested in tests/voiceCommands.test.ts.
 * Depends : nothing.
 * Question you should be able to answer: why are "pause" and "resume" handled here and never sent to the server?
 */
export type Command =
  | { kind: "ask"; question: string }
  | { kind: "pause" }
  | { kind: "resume" }
  | { kind: "describe" }
  | { kind: "read" }
  | { kind: "empty" };

export const TRIGGER_WORD = "vision";

const PAUSE_WORDS = ["pause", "stop", "quiet", "mute"];
const RESUME_WORDS = ["resume", "continue", "start", "unmute"];

export function parseCommand(transcript: string, trigger: string = TRIGGER_WORD): Command | null {
  const text = transcript.toLowerCase();
  const idx = text.indexOf(trigger);
  if (idx === -1) return null;
  const after = text
    .slice(idx + trigger.length)
    .replace(/^[\s,.:;!?-]+/, "")   // strip "vision," / "vision:" punctuation
    .replace(/[.?!,]+$/, "")        // strip trailing punctuation
    .trim();
  if (after === "") return { kind: "empty" };
  const first = after.split(/\s+/)[0];
  if (PAUSE_WORDS.includes(first)) return { kind: "pause" };
  if (RESUME_WORDS.includes(first)) return { kind: "resume" };
  if (after === "describe") return { kind: "describe" };
  if (after === "read") return { kind: "read" };
  return { kind: "ask", question: after };
}
