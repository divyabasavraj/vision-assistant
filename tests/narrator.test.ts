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

describe("class-aware distance", () => {
  it("calls a curb close while the same box height is still far for a person", () => {
    expect(bucketOf(0.08, NARRATOR_CONFIG, "curb")).toBe("close");
    expect(bucketOf(0.08, NARRATOR_CONFIG, "person")).toBe("far");
  });

  it("makes a bus fill more of the frame than a person before it counts as close", () => {
    expect(bucketOf(0.6, NARRATOR_CONFIG, "person")).toBe("close");
    expect(bucketOf(0.6, NARRATOR_CONFIG, "bus")).toBe("near");
  });

  it("treats an unknown label as person-sized", () => {
    expect(bucketOf(0.6, NARRATOR_CONFIG, "spaceship")).toBe("close");
  });
});

describe("narrator events", () => {
  it("says nothing until an object is seen 3 frames in a row, then announces it once", () => {
    const n = createNarrator();
    expect(n.update([box("person", 0.2, 0.3)], 0)).toBeNull();
    expect(n.update([box("person", 0.2, 0.3)], 100)).toBeNull();
    expect(n.update([box("person", 0.2, 0.3)], 200)).toEqual({ text: "person, left, near", priority: 4, interrupt: false, zone: "left" });
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
    // Smoothing means "close" takes two frames of real growth, not one jump. That is the point of it.
    expect(n.update([box("person", 0.2, 0.7)], 2000)).toBeNull();
    expect(n.update([box("person", 0.2, 0.75)], 2100)).toEqual({ text: "person, left, close", priority: 2, interrupt: false, zone: "left" });
  });

  it("rate-limits to one phrase per 1.5 s but lets priority 1 interrupt", () => {
    const n = createNarrator();
    const both = [box("chair", 0.2, 0.2, 1), box("table", 0.8, 0.2, 2)];
    for (let t = 0; t < 200; t += 100) n.update(both, t);
    expect(n.update(both, 200)?.text).toBe("chair, left, near");
    expect(n.update(both, 300)).toBeNull();                              // table waits
    expect(n.update(both, 1800)?.text).toBe("table, right, near");        // 1.5 s later it speaks
    const pole = [...both, box("pole", 0.5, 0.85, 3)];
    for (let t = 1900; t < 2100; t += 100) n.update(pole, t);
    expect(n.update(pole, 2100)).toEqual({ text: "pole, ahead, close", priority: 1, interrupt: true, zone: "ahead" });
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
    for (let t = 0; t < 300; t += 100) n.update([box("pole", 0.5, 0.85)], t);
    let out: Phrase | null = null;
    for (let f = 0; f <= NARRATOR_CONFIG.dropAfterFrames; f++) out = n.update([], 3000 + f * 100);  // 11 empty frames -> missed > 10
    expect(out).toEqual({ text: "path clear", priority: 4, interrupt: false, zone: "ahead" });
  });
});

describe("far objects off to the side", () => {
  it("stays silent about a far object that is not straight ahead", () => {
    const n = createNarrator();
    for (let t = 0; t < 500; t += 100) expect(n.update([box("person", 0.1, 0.1)], t)).toBeNull();
  });

  it("still announces a far object straight ahead", () => {
    const n = createNarrator();
    n.update([box("person", 0.5, 0.1)], 0);
    n.update([box("person", 0.5, 0.1)], 100);
    expect(n.update([box("person", 0.5, 0.1)], 200)?.text).toBe("person, ahead, far");
  });
});

describe("height smoothing", () => {
  it("does not jump to close on a single-frame height spike", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("person", 0.5, 0.24)], t);
    // Raw height 0.52 says "close"; the smoothed height (0.352) says "near", and near is what gets spoken.
    expect(n.update([box("person", 0.5, 0.52)], 2300)?.text).toBe("person, ahead, near");
  });

  it("treats an impossible one-frame size jump as a different object, not an approach", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("person", 0.5, 0.3)], t);
    // 0.3 -> 0.9 is IoU 0.33, below the 0.4 match gate, so this is a new unconfirmed track.
    expect(n.update([box("person", 0.5, 0.9)], 2000)).toBeNull();
  });
});

describe("fast approach", () => {
  it("warns before the object is close when it is growing quickly", () => {
    const n = createNarrator();
    const said: string[] = [];
    [0.10, 0.13, 0.17, 0.22, 0.29].forEach((h, i) => {
      const p = n.update([box("car", 0.5, h)], i * 100);
      if (p) said.push(p.text);
    });
    expect(said).toContain("car approaching fast, ahead");
  });

  it("interrupts, because a fast approach is priority 1", () => {
    const n = createNarrator();
    let urgent: Phrase | null = null;
    [0.10, 0.13, 0.17, 0.22, 0.29].forEach((h, i) => {
      const p = n.update([box("car", 0.5, h)], i * 100);
      if (p?.text.includes("approaching fast")) urgent = p;
    });
    expect(urgent).toEqual({ text: "car approaching fast, ahead", priority: 1, interrupt: true, zone: "ahead" });
  });

  it("stays quiet when the object grows slowly", () => {
    const n = createNarrator();
    const said: string[] = [];
    [0.10, 0.11, 0.12, 0.13, 0.14].forEach((h, i) => {
      const p = n.update([box("car", 0.5, h)], i * 1000);   // one frame per second
      if (p) said.push(p.text);
    });
    expect(said.some((s) => s.includes("approaching fast"))).toBe(false);
  });

  it("ignores a fast-growing object that is off to the side", () => {
    const n = createNarrator();
    const said: string[] = [];
    [0.10, 0.13, 0.17, 0.22, 0.29].forEach((h, i) => {
      const p = n.update([box("car", 0.1, h)], i * 100);
      if (p) said.push(p.text);
    });
    expect(said.some((s) => s.includes("approaching fast"))).toBe(false);
  });
});
