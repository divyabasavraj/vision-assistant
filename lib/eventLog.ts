/**
 * Purpose : Evaluation log (Task 11). Every spoken phrase and every Q&A round-trip, with a timestamp,
 *           so narration and latency numbers in the report come from the app itself, not a stopwatch.
 * Depends : nothing — the clock is passed in so tests can control time.
 * Question you should be able to answer: why is the log copied to the clipboard instead of uploaded?
 */
export type LogKind = "phrase" | "ask" | "answer" | "error" | "stt-restart";

export type LogEntry = {
  t: number;              // ms since the log was created
  kind: LogKind;
  text: string;
  ms?: number;            // Q&A round-trip
  provider?: string;      // which cloud model answered
  backend?: string;       // detector backend at the time
  fps?: number;           // detector fps at the time
};

/** Keeps the newest `max` entries, so a forgotten session cannot grow without bound. */
export function createEventLog(now: () => number, max = 2000) {
  const t0 = now();
  const entries: LogEntry[] = [];
  return {
    entries: entries as readonly LogEntry[],
    record(kind: LogKind, text: string, extra: Omit<LogEntry, "t" | "kind" | "text"> = {}): void {
      entries.push({ t: Math.round(now() - t0), kind, text, ...extra });
      if (entries.length > max) entries.shift();
    },
    toJSON(meta: Record<string, string> = {}): string {
      return JSON.stringify({ ...meta, log: entries }, null, 2);
    },
  };
}
export type EventLog = ReturnType<typeof createEventLog>;
