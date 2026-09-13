/**
 * Purpose : Decide WHEN to speak and WHAT to say from a stream of boxes. Pure logic, no browser.
 *           Tracks objects across frames (IoU matching), confirms after 3 frames, speaks only on events
 *           (new / closer / in path / fast approach / clear), one phrase per 1.5 s, priority-1 may interrupt.
 * Depends : Box type from lib/detector.ts. Tested in tests/narrator.test.ts.
 * Question you should be able to answer: why do we track objects instead of speaking every frame?
 */
import type { Box } from "./detector";
import { iou, type RawBox } from "./detector";

export const NARRATOR_CONFIG = {
  matchIou: 0.4,            // same-class boxes overlapping at least this much are the same object
  confirmFrames: 3,         // consecutive frames before an object is real (kills flicker)
  dropAfterFrames: 10,      // consecutive misses before an object is forgotten
  minGapMs: 1500,           // at most one phrase per this interval (priority 1 ignores it)
  repeatSameObjectMs: 5000, // same object not re-announced within this unless it got closer
  nearHeight: 0.25,         // box height as fraction of frame: < near = far, < close = near, else close
  closeHeight: 0.5,
  smoothAlpha: 0.4,         // EMA weight on the newest height; one noisy frame must not move a bucket
  ttcWarnSec: 1.5,          // time-to-contact below this, straight ahead = urgent
  approachRepeatMs: 2000,   // don't repeat the same fast-approach warning more often than this
};
export type NarratorConfig = typeof NARRATOR_CONFIG;
export type Zone = "left" | "ahead" | "right";
export type Bucket = "far" | "near" | "close";
export type Phrase = { text: string; priority: 1 | 2 | 3 | 4; interrupt: boolean; zone: Zone };

const MOVING = new Set(["person", "dog", "car", "bicycle", "motorbike", "auto", "bus"]);
const LEVEL_CHANGE = new Set(["stairs_up", "stairs_down", "curb"]);
/** Things you walk towards, not around. They never count as occupying a side. */
const GUIDES = new Set(["crosswalk", "door"]);
const BUCKET_RANK: Record<Bucket, number> = { far: 0, near: 1, close: 2 };

/**
 * How tall each class is in the real world, relative to a person. Box height alone is a bad distance
 * cue: a curb fills almost none of the frame even at your feet, a bus fills it from far away. Scaling
 * the far/near thresholds by this makes "close" mean roughly the same distance for every class.
 * Ground-level classes (curb, crosswalk) are the weakest case — see the note in the design doc.
 */
const CLASS_SIZE: Record<string, number> = {
  person: 1.0, dog: 0.35, car: 0.9, bicycle: 0.65, motorbike: 0.75, auto: 1.0, bus: 1.8,
  stairs_up: 0.6, stairs_down: 0.6, curb: 0.1, pole: 1.5, chair: 0.55, table: 0.45,
  door: 1.2, crosswalk: 0.15,
};
export function sizeScale(label: string): number { return CLASS_SIZE[label] ?? 1; }

export function zoneOf(cx: number): Zone { return cx < 1 / 3 ? "left" : cx < 2 / 3 ? "ahead" : "right"; }
export function bucketOf(h: number, cfg: NarratorConfig, label = ""): Bucket {
  const s = sizeScale(label);
  return h < cfg.nearHeight * s ? "far" : h < cfg.closeHeight * s ? "near" : "close";
}
/** Fixed shape "<what>, <where>, <how far>"; a far stairs/curb is still worth a heads-up but "far" adds nothing. */
export function phraseFor(label: string, zone: Zone, bucket: Bucket): string {
  const name = label.replace(/_/g, " ");
  if (LEVEL_CHANGE.has(label) && bucket === "far") return `${name}, ${zone}`;
  return `${name}, ${zone}, ${bucket}`;
}
/** Deliberately different wording so an urgent warning never sounds like routine narration. */
export function approachPhraseFor(label: string): string {
  return `${label.replace(/_/g, " ")} approaching fast, ahead`;
}

type Track = {
  id: number; label: string; box: Box;
  seen: number; missed: number; confirmed: boolean;
  smoothH: number;          // EMA of box height — every distance decision reads this, never box.h
  prevH: number; prevT: number; updatedAt: number;   // previous smoothed sample, for the growth rate
  spokenAt: number | null; spokenBucket: Bucket | null; spokenZone: Zone | null;
  spokenApproachAt: number | null;
};
type Candidate = { track: Track; phrase: Phrase; bucket: Bucket; zone: Zone; approach: boolean };

const toRaw = (b: Box): RawBox => ({ cls: b.cls, score: b.score, x1: b.cx - b.w / 2, y1: b.cy - b.h / 2, x2: b.cx + b.w / 2, y2: b.cy + b.h / 2 });

export function createNarrator(cfg: NarratorConfig = NARRATOR_CONFIG) {
  let tracks: Track[] = [];
  let nextId = 1;
  let lastPhraseAt = -Infinity;
  let pendingClear = false;

  /** Greedy per-track matching: each existing track takes the best-overlapping unclaimed box of its class. */
  function match(boxes: Box[], now: number) {
    const unmatched = new Set(boxes.map((_, i) => i));
    for (const t of tracks) {
      let best = -1, bestIou = cfg.matchIou;
      for (const i of unmatched) {
        if (boxes[i].label !== t.label) continue;
        const v = iou(toRaw(t.box), toRaw(boxes[i]));
        if (v >= bestIou) { bestIou = v; best = i; }
      }
      if (best >= 0) {
        t.box = boxes[best];
        t.prevH = t.smoothH; t.prevT = t.updatedAt;
        t.smoothH = cfg.smoothAlpha * boxes[best].h + (1 - cfg.smoothAlpha) * t.smoothH;
        t.updatedAt = now;
        t.seen += 1; t.missed = 0;
        unmatched.delete(best);
      } else { t.missed += 1; }
    }
    // An unconfirmed track that misses even one frame is thrown away: "3 consecutive frames" means consecutive.
    tracks = tracks.filter((t) => t.confirmed || t.missed === 0);
    for (const i of unmatched) {
      const b = boxes[i];
      tracks.push({ id: nextId++, label: b.label, box: b, seen: 1, missed: 0, confirmed: false, smoothH: b.h, prevH: b.h, prevT: now, updatedAt: now, spokenAt: null, spokenBucket: null, spokenZone: null, spokenApproachAt: null });
    }
    const dropped = tracks.filter((t) => t.missed > cfg.dropAfterFrames);
    if (dropped.some((t) => t.spokenZone === "ahead" && t.spokenBucket === "close")) pendingClear = true;
    tracks = tracks.filter((t) => t.missed <= cfg.dropAfterFrames);
  }

  /**
   * Monocular time-to-contact: a box growing at rate dh/dt reaches you in about h / (dh/dt) seconds.
   * No depth sensor needed — only how fast the thing is filling the frame. Infinity = not approaching.
   */
  function timeToContact(t: Track): number {
    const dt = (t.updatedAt - t.prevT) / 1000;
    if (dt <= 0) return Infinity;
    const rate = (t.smoothH - t.prevH) / dt;
    return rate > 0 ? t.smoothH / rate : Infinity;
  }

  const onScreen = () => tracks.filter((t) => t.confirmed && t.missed === 0 && !GUIDES.has(t.label));
  /** A side is unusable if anything near or close sits in it; something far away won't stop you stepping across. */
  const sideBlocked = (zone: Zone) => onScreen().some((t) => zoneOf(t.box.cx) === zone && bucketOf(t.smoothH, cfg, t.label) !== "far");
  const sideCount = (zone: Zone) => onScreen().filter((t) => zoneOf(t.box.cx) === zone).length;

  /**
   * Where to go when the way ahead is blocked. Returns null for level changes (you cannot side-step a
   * step — it is still there, and steering someone sideways beside stairs is worse than not advising)
   * and for guides: a door ahead is where you are trying to go, so routing around it is backwards.
   * With both sides open, the emptier one wins; an exact tie always resolves left so it cannot dither.
   */
  function escape(blockerLabel: string): string | null {
    if (LEVEL_CHANGE.has(blockerLabel) || GUIDES.has(blockerLabel)) return null;
    const left = sideBlocked("left"), right = sideBlocked("right");
    if (left && right) return "stop";
    if (left) return "go right";
    if (right) return "go left";
    return sideCount("right") < sideCount("left") ? "go right" : "go left";
  }

  function priorityOf(t: Track, zone: Zone, bucket: Bucket, closer: boolean): 1 | 2 | 3 | 4 {
    if (zone === "ahead" && bucket === "close") return 1;
    if (MOVING.has(t.label) && closer) return 2;
    if (LEVEL_CHANGE.has(t.label)) return 3;
    return 4;
  }

  /** Confirmed, currently-visible tracks that have something new to say. */
  function candidates(now: number): Candidate[] {
    const out: Candidate[] = [];
    for (const t of tracks) {
      if (t.missed > 0) continue;
      if (!t.confirmed && t.seen >= cfg.confirmFrames) t.confirmed = true;
      if (!t.confirmed) continue;
      const zone = zoneOf(t.box.cx), bucket = bucketOf(t.smoothH, cfg, t.label);
      // Something far away and off to one side is not your problem — only the middle third is.
      if (bucket === "far" && zone !== "ahead") continue;

      const cooling = t.spokenApproachAt !== null && now - t.spokenApproachAt < cfg.approachRepeatMs;
      if (zone === "ahead" && !cooling && timeToContact(t) < cfg.ttcWarnSec) {
        out.push({ track: t, zone, bucket, approach: true, phrase: { text: approachPhraseFor(t.label), priority: 1, interrupt: true, zone } });
        continue;
      }

      const isNew = t.spokenAt === null;
      const closer = t.spokenBucket !== null && BUCKET_RANK[bucket] > BUCKET_RANK[t.spokenBucket];
      const inPath = zone === "ahead" && t.spokenZone !== null && t.spokenZone !== "ahead";
      if (!isNew && !closer && !inPath) continue;
      if (!isNew && !closer && t.spokenAt !== null && now - t.spokenAt < cfg.repeatSameObjectMs) continue;
      const priority = priorityOf(t, zone, bucket, closer);
      out.push({ track: t, zone, bucket, approach: false, phrase: { text: phraseFor(t.label, zone, bucket), priority, interrupt: priority === 1, zone } });
    }
    return out;
  }

  return {
    update(boxes: Box[], nowMs: number): Phrase | null {
      match(boxes, nowMs);
      const cands = candidates(nowMs).sort((a, b) => a.phrase.priority - b.phrase.priority);
      const gapOk = nowMs - lastPhraseAt >= cfg.minGapMs;
      const pick = cands[0];
      if (pick && (gapOk || pick.phrase.priority === 1)) {
        // Only a priority-1 phrase means "genuinely blocked", and only then is a direction useful.
        if (pick.phrase.priority === 1) {
          const way = escape(pick.track.label);
          if (way) pick.phrase.text = `${pick.phrase.text}, ${way}`;
        }
        pick.track.spokenAt = nowMs; pick.track.spokenBucket = pick.bucket; pick.track.spokenZone = pick.zone;
        if (pick.approach) pick.track.spokenApproachAt = nowMs;
        lastPhraseAt = nowMs;
        pendingClear = false;
        return pick.phrase;
      }
      if (pendingClear && gapOk) {
        const visible = tracks.filter((t) => t.confirmed && t.missed === 0);
        const stillBlocked = visible.some((t) => zoneOf(t.box.cx) === "ahead" && bucketOf(t.smoothH, cfg, t.label) === "close");
        pendingClear = false;
        // Only claim the centre is clear while we can still see something to be clear of. An empty
        // frame means the camera stopped seeing, which is not the same as the obstacle going away —
        // the phone may simply have turned. And the phrase says "ahead", not "path": all we know is
        // what is in the middle third of one frame.
        if (!stillBlocked && visible.length > 0) { lastPhraseAt = nowMs; return { text: "ahead clear", priority: 4, interrupt: false, zone: "ahead" }; }
      }
      return null;
    },
    reset() { tracks = []; lastPhraseAt = -Infinity; pendingClear = false; },
  };
}
