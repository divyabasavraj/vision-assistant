import { describe, it, expect } from "vitest";
import { createNarrator, zoneOf, bucketOf, phraseFor, NARRATOR_CONFIG, type Phrase } from "../lib/narrator";
import type { Box } from "../lib/detector";

const box = (label: string, cx: number, h: number, cls = 0): Box => ({ cls, label, score: 0.9, cx, cy: 0.5, w: 0.2, h });

describe("helpers", () => {
  it("zones by thirds", () => { expect(zoneOf(0.1)).toBe("left"); expect(zoneOf(0.5)).toBe("ahead"); expect(zoneOf(0.9)).toBe("right"); });
  it("buckets by height", () => { expect(bucketOf(0.1, NARRATOR_CONFIG)).toBe("far"); expect(bucketOf(0.3, NARRATOR_CONFIG)).toBe("near"); expect(bucketOf(0.7, NARRATOR_CONFIG)).toBe("close"); });
  it("formats phrases", () => {
    expect(phraseFor("person", "left", "near")).toBe("person, left, near");
    expect(phraseFor("stairs_down", "ahead", "far")).toBe("stairs down, ahead");   // level changes omit "far"
    expect(phraseFor("stairs_down", "ahead", "close")).toBe("stairs down, ahead, close");
  });
});

describe("narrator events", () => {
  it("says nothing until an object is seen 3 frames in a row, then announces it once", () => {
    const n = createNarrator();
    expect(n.update([box("person", 0.2, 0.3)], 0)).toBeNull();
    expect(n.update([box("person", 0.2, 0.3)], 100)).toBeNull();
    expect(n.update([box("person", 0.2, 0.3)], 200)).toEqual({ text: "person, left, near", priority: 4, interrupt: false });
    expect(n.update([box("person", 0.2, 0.3)], 300)).toBeNull();
  });

  it("filters a one-frame flicker", () => {
    const n = createNarrator();
    n.update([box("dog", 0.5, 0.3)], 0);
    n.update([], 100);
    n.update([box("dog", 0.5, 0.3)], 200);
    expect(n.update([box("dog", 0.5, 0.3)], 300)).toBeNull();   // only 2 consecutive frames so far
  });

  it("announces again when the object gets closer, with priority 2 for moving classes", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("person", 0.2, 0.3)], t);
    expect(n.update([box("person", 0.2, 0.6)], 2000)).toEqual({ text: "person, left, close", priority: 2, interrupt: false });
  });

  it("rate-limits to one phrase per 1.5 s but lets priority 1 interrupt", () => {
    const n = createNarrator();
    const both = [box("chair", 0.2, 0.3, 1), box("table", 0.8, 0.3, 2)];
    for (let t = 0; t < 200; t += 100) n.update(both, t);
    expect(n.update(both, 200)?.text).toBe("chair, left, near");
    expect(n.update(both, 300)).toBeNull();                              // table waits
    expect(n.update(both, 1800)?.text).toBe("table, right, near");        // 1.5 s later it speaks
    const pole = [...both, box("pole", 0.5, 0.7, 3)];
    for (let t = 1900; t < 2100; t += 100) n.update(pole, t);
    expect(n.update(pole, 2100)).toEqual({ text: "pole, ahead, close", priority: 1, interrupt: true });
  });

  it("does not repeat the same object within 5 s unless closer", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("person", 0.2, 0.3)], t);   // announced at t=200
    // Drift into the middle third in steps small enough to keep IoU >= 0.4 (a teleport would be a new object).
    for (let t = 400, cx = 0.2; cx < 0.5; t += 100, cx += 0.06) expect(n.update([box("person", cx, 0.3)], t)).toBeNull();
    expect(n.update([box("person", 0.5, 0.3)], 2000)).toBeNull();                 // now "ahead" but < 5 s since spoken
    expect(n.update([box("person", 0.5, 0.3)], 6000)?.text).toBe("person, ahead, near");
  });

  it("says path clear when an ahead-close object disappears", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("pole", 0.5, 0.7)], t);
    let out: Phrase | null = null;
    for (let f = 0; f <= NARRATOR_CONFIG.dropAfterFrames; f++) out = n.update([], 3000 + f * 100);  // 11 empty frames -> missed > 10
    expect(out).toEqual({ text: "path clear", priority: 4, interrupt: false });
  });
});
