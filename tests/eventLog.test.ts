import { describe, it, expect } from "vitest";
import { createEventLog } from "../lib/eventLog";

describe("createEventLog", () => {
  it("stamps entries relative to creation time", () => {
    let clock = 1000;
    const log = createEventLog(() => clock);
    clock = 1250.4;
    log.record("phrase", "person ahead", { fps: 5 });
    clock = 3000;
    log.record("answer", "a red chair", { ms: 1800, provider: "groq" });
    expect(log.entries).toEqual([
      { t: 250, kind: "phrase", text: "person ahead", fps: 5 },
      { t: 2000, kind: "answer", text: "a red chair", ms: 1800, provider: "groq" },
    ]);
  });
  it("drops the oldest entries past the cap", () => {
    const log = createEventLog(() => 0, 3);
    for (const s of ["a", "b", "c", "d"]) log.record("phrase", s);
    expect(log.entries.map((e) => e.text)).toEqual(["b", "c", "d"]);
  });
  it("exports JSON with metadata", () => {
    const log = createEventLog(() => 0);
    log.record("ask", "what is this");
    const out = JSON.parse(log.toJSON({ userAgent: "test" }));
    expect(out.userAgent).toBe("test");
    expect(out.log).toHaveLength(1);
  });
});
