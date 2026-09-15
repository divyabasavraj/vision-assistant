import { describe, it, expect } from "vitest";
import { createNarrator, zoneOf, bucketOf, phraseFor, NARRATOR_CONFIG, type Phrase } from "../lib/narrator";
import type { Box } from "../lib/detector";

const box = (label: string, cx: number, h: number, cls = 0): Box => ({ cls, label, score: 0.9, cx, cy: 0.5, w: 0.2, h });

describe("helpers", () => {
  it("zones by thirds", () => { expect(zoneOf(0.1)).toBe("left"); expect(zoneOf(0.5)).toBe("ahead"); expect(zoneOf(0.9)).toBe("right"); });
  it("buckets by height", () => { expect(bucketOf(0.1, NARRATOR_CONFIG)).toBe("far"); expect(bucketOf(0.3, NARRATOR_CONFIG)).toBe("near"); expect(bucketOf(0.7, NARRATOR_CONFIG)).toBe("close"); });
  it("formats phrases", () => {
    expect(phraseFor("person", "left", "near")).toBe("person on your left, a few steps away");
    expect(phraseFor("person", "left", "close")).toBe("person right next to you, on the left");
    expect(phraseFor("person", "ahead", "far")).toBe("person further ahead");
    expect(phraseFor("person", "ahead", "near")).toBe("person a few steps ahead");
    expect(phraseFor("person", "ahead", "close")).toBe("person right in front of you");
    expect(phraseFor("stairs_down", "ahead", "far")).toBe("stairs going down ahead");   // level changes omit distance
    expect(phraseFor("stairs_down", "ahead", "close")).toBe("stairs going down right in front of you");
  });

  // Six labels do not say themselves out loud. Bare "auto" is meaningless spoken, and an
  // underscore read literally ("stairs underscore up") is worse than useless.
  it("speaks label names the way a person would say them", () => {
    expect(phraseFor("auto", "ahead", "near")).toBe("auto rickshaw a few steps ahead");
    expect(phraseFor("crosswalk", "ahead", "near")).toBe("crossing a few steps ahead");
    expect(phraseFor("stairs_up", "ahead", "close")).toBe("stairs going up right in front of you");
  });

  // The narrator filters far side-objects out before speaking, so this never reaches the ear today.
  // It still must not claim "a few steps away" about something far off, or the fallback lies the day
  // that filter changes.
  it("does not call a far side-object a few steps away", () => {
    expect(phraseFor("person", "right", "far")).toBe("person over on your right");
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
    expect(n.update([box("person", 0.2, 0.3)], 200)).toEqual({ text: "person on your left, a few steps away", priority: 4, interrupt: false, zone: "left" });
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
    expect(n.update([box("person", 0.2, 0.75)], 2100)).toEqual({ text: "person right next to you, on the left", priority: 2, interrupt: false, zone: "left" });
  });

  it("rate-limits to one phrase per 1.5 s but lets priority 1 interrupt", () => {
    const n = createNarrator();
    const both = [box("chair", 0.2, 0.2, 1), box("table", 0.8, 0.2, 2)];
    for (let t = 0; t < 200; t += 100) n.update(both, t);
    expect(n.update(both, 200)?.text).toBe("chair on your left, a few steps away");
    expect(n.update(both, 300)).toBeNull();                              // table waits
    expect(n.update(both, 1800)?.text).toBe("table on your right, a few steps away");        // 1.5 s later it speaks
    const pole = [...both, box("pole", 0.5, 0.85, 3)];
    for (let t = 1900; t < 2100; t += 100) n.update(pole, t);
    // chair blocks the left and table blocks the right, so there is genuinely nowhere to step.
    expect(n.update(pole, 2100)).toEqual({ text: "pole right in front of you, stop", priority: 1, interrupt: true, zone: "ahead" });
  });

  it("does not repeat the same object within 5 s unless closer", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("person", 0.2, 0.3)], t);   // announced at t=200
    // Drift into the middle third in steps small enough to keep IoU >= 0.4 (a teleport would be a new object).
    for (let t = 400, cx = 0.2; cx < 0.5; t += 100, cx += 0.06) expect(n.update([box("person", cx, 0.3)], t)).toBeNull();
    expect(n.update([box("person", 0.5, 0.3)], 2000)).toBeNull();                 // now "ahead" but < 5 s since spoken
    expect(n.update([box("person", 0.5, 0.3)], 6000)?.text).toBe("person a few steps ahead");
  });

  it("says ahead is clear when the blocker goes but the scene is still in view", () => {
    const n = createNarrator();
    const chair = box("chair", 0.1, 0.2, 1);
    for (let t = 0; t < 300; t += 100) n.update([box("pole", 0.5, 0.85, 0), chair], t);
    let out: Phrase | null = null;
    for (let f = 0; f <= NARRATOR_CONFIG.dropAfterFrames; f++) out = n.update([chair], 3000 + f * 1000);
    expect(out).toEqual({ text: "ahead is clear", priority: 4, interrupt: false, zone: "ahead" });
  });

  it("stays silent when everything vanishes at once, because that means the camera moved", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("pole", 0.5, 0.85)], t);
    // An empty frame is not evidence of a clear path. The pole may still be right in front of you
    // and the phone simply turned away from it.
    let out: Phrase | null = null;
    for (let f = 0; f <= NARRATOR_CONFIG.dropAfterFrames; f++) out = n.update([], 3000 + f * 1000);
    expect(out).toBeNull();
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
    expect(n.update([box("person", 0.5, 0.1)], 200)?.text).toBe("person further ahead");
  });
});

describe("height smoothing", () => {
  it("does not jump to close on a single-frame height spike", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("person", 0.5, 0.24)], t);
    // Raw height 0.52 says "close"; the smoothed height (0.352) says "near", and near is what gets spoken.
    expect(n.update([box("person", 0.5, 0.52)], 2300)?.text).toBe("person a few steps ahead");
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
    expect(said).toContain("car coming at you fast, go left");
  });

  it("interrupts, because a fast approach is priority 1", () => {
    const n = createNarrator();
    let urgent: Phrase | null = null;
    [0.10, 0.13, 0.17, 0.22, 0.29].forEach((h, i) => {
      const p = n.update([box("car", 0.5, h)], i * 100);
      if (p?.text.includes("coming at you fast")) urgent = p;
    });
    expect(urgent).toEqual({ text: "car coming at you fast, go left", priority: 1, interrupt: true, zone: "ahead" });
  });

  it("stays quiet when the object grows slowly", () => {
    const n = createNarrator();
    const said: string[] = [];
    [0.10, 0.11, 0.12, 0.13, 0.14].forEach((h, i) => {
      const p = n.update([box("car", 0.5, h)], i * 1000);   // one frame per second
      if (p) said.push(p.text);
    });
    expect(said.some((s) => s.includes("coming at you fast"))).toBe(false);
  });

  it("ignores a fast-growing object that is off to the side", () => {
    const n = createNarrator();
    const said: string[] = [];
    [0.10, 0.13, 0.17, 0.22, 0.29].forEach((h, i) => {
      const p = n.update([box("car", 0.1, h)], i * 100);
      if (p) said.push(p.text);
    });
    expect(said.some((s) => s.includes("coming at you fast"))).toBe(false);
  });

  // Field test, 2026-09-15: a person standing about a metre away walked off, and the phone said
  // "person coming at you fast". At that range a portrait phone cannot fit a whole person, so the
  // detector boxes only the part it can see. As they back away, more of them comes into frame and the
  // box GROWS for a few frames — which is exactly what an approach looks like to a height-only
  // time-to-contact. A box touching the top or bottom edge has a height set by the frame, not by the
  // distance, so it must never be evidence of approach.
  it("does not call a person backing out of a clipped frame an approach", () => {
    const n = createNarrator();
    const said: string[] = [];
    // bottom-anchored: feet are cut off by the frame edge (cy = 1 - h/2), 5 fps
    [0.45, 0.45, 0.45, 0.70, 0.95, 0.90, 0.80].forEach((h, i) => {
      const p = n.update([{ cls: 0, label: "person", score: 0.9, cx: 0.5, cy: 1 - h / 2, w: 0.3, h }], i * 200);
      if (p) said.push(p.text);
    });
    expect(said.some((s) => s.includes("coming at you fast"))).toBe(false);
  });

  it("does not trust growth measured from a box that was clipped on the previous frame either", () => {
    const n = createNarrator();
    const said: string[] = [];
    // head cut off at the top edge (cy = h/2), then the whole person fits and the box jumps up in one frame
    const frames: Array<[number, number]> = [[0.5, 0.25], [0.5, 0.25], [0.5, 0.25], [0.9, 0.5], [0.85, 0.5], [0.8, 0.5]];
    frames.forEach(([h, cy], i) => {
      const p = n.update([{ cls: 0, label: "person", score: 0.9, cx: 0.5, cy, w: 0.3, h }], i * 200);
      if (p) said.push(p.text);
    });
    expect(said.some((s) => s.includes("coming at you fast"))).toBe(false);
  });
});

describe("escape direction", () => {
  const confirm = (n: ReturnType<typeof createNarrator>, boxes: Box[]) => {
    let out: Phrase | null = null;
    for (let t = 0; t < 300; t += 100) out = n.update(boxes, t);
    return out;
  };

  it("tells you which way to go when something close blocks the path", () => {
    const n = createNarrator();
    expect(confirm(n, [box("pole", 0.5, 0.85)])?.text).toBe("pole right in front of you, go left");
  });

  it("sends you to the clear side when one side is occupied", () => {
    const n = createNarrator();
    expect(confirm(n, [box("pole", 0.5, 0.85, 0), box("person", 0.15, 0.3, 1)])?.text)
      .toBe("pole right in front of you, go right");
  });

  it("says stop when both sides are blocked too", () => {
    const n = createNarrator();
    expect(confirm(n, [box("pole", 0.5, 0.85, 0), box("person", 0.15, 0.3, 1), box("person", 0.85, 0.3, 1)])?.text)
      .toBe("pole right in front of you, stop");
  });

  it("does not tell you to walk around a door, which is where you are trying to go", () => {
    const n = createNarrator();
    expect(confirm(n, [box("door", 0.5, 0.7)])?.text).toBe("door right in front of you");
  });

  it("never tells you to step around a level change, because you cannot", () => {
    const n = createNarrator();
    expect(confirm(n, [box("stairs_down", 0.5, 0.4)])?.text).toBe("stairs going down right in front of you");
  });

  it("does not add a direction to routine, non-blocking phrases", () => {
    const n = createNarrator();
    expect(confirm(n, [box("person", 0.2, 0.3)])?.text).toBe("person on your left, a few steps away");
  });

  it("does not treat a crosswalk as something blocking a side", () => {
    const n = createNarrator();
    expect(confirm(n, [box("pole", 0.5, 0.85, 0), box("crosswalk", 0.15, 0.1, 1)])?.text)
      .toBe("pole right in front of you, go left");
  });

  it("does not let a far object block a side", () => {
    const n = createNarrator();
    expect(confirm(n, [box("pole", 0.5, 0.85, 0), box("person", 0.15, 0.1, 1), box("person", 0.85, 0.3, 2)])?.text)
      .toBe("pole right in front of you, go left");
  });

  it("gives a way out of a fast approach as well", () => {
    const n = createNarrator();
    let urgent: Phrase | null = null;
    [0.10, 0.13, 0.17, 0.22].forEach((h, i) => {
      const p = n.update([box("car", 0.5, h)], i * 100);
      if (p?.text.includes("coming at you fast")) urgent = p;
    });
    expect(urgent).toEqual({ text: "car coming at you fast, go left", priority: 1, interrupt: true, zone: "ahead" });
  });
});
