/**
 * Purpose : Decide WHEN to speak and WHAT to say from a stream of boxes. Pure logic, no browser.
 *           Tracks objects across frames (IoU matching), confirms after 3 frames, speaks only on events
 *           (new / closer / in path / clear), one phrase per 1.5 s, priority-1 may interrupt.
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
};
export type NarratorConfig = typeof NARRATOR_CONFIG;
export type Zone = "left" | "ahead" | "right";
export type Bucket = "far" | "near" | "close";
export type Phrase = { text: string; priority: 1 | 2 | 3 | 4; interrupt: boolean };

const MOVING = new Set(["person", "dog", "car", "bicycle", "motorbike", "auto", "bus"]);
const LEVEL_CHANGE = new Set(["stairs_up", "stairs_down", "curb"]);
const BUCKET_RANK: Record<Bucket, number> = { far: 0, near: 1, close: 2 };

export function zoneOf(cx: number): Zone { return cx < 1 / 3 ? "left" : cx < 2 / 3 ? "ahead" : "right"; }
export function bucketOf(h: number, cfg: NarratorConfig): Bucket { return h < cfg.nearHeight ? "far" : h < cfg.closeHeight ? "near" : "close"; }
/** Fixed shape "<what>, <where>, <how far>"; a far stairs/curb is still worth a heads-up but "far" adds nothing. */
export function phraseFor(label: string, zone: Zone, bucket: Bucket): string {
  const name = label.replace(/_/g, " ");
  if (LEVEL_CHANGE.has(label) && bucket === "far") return `${name}, ${zone}`;
  return `${name}, ${zone}, ${bucket}`;
}

type Track = {
  id: number; label: string; box: Box;
  seen: number; missed: number; confirmed: boolean;
  spokenAt: number | null; spokenBucket: Bucket | null; spokenZone: Zone | null;
};
type Candidate = { track: Track; phrase: Phrase; bucket: Bucket; zone: Zone };

const toRaw = (b: Box): RawBox => ({ cls: b.cls, score: b.score, x1: b.cx - b.w / 2, y1: b.cy - b.h / 2, x2: b.cx + b.w / 2, y2: b.cy + b.h / 2 });

export function createNarrator(cfg: NarratorConfig = NARRATOR_CONFIG) {
  let tracks: Track[] = [];
  let nextId = 1;
  let lastPhraseAt = -Infinity;
  let pendingClear = false;

  /** Greedy per-track matching: each existing track takes the best-overlapping unclaimed box of its class. */
  function match(boxes: Box[]) {
    const unmatched = new Set(boxes.map((_, i) => i));
    for (const t of tracks) {
      let best = -1, bestIou = cfg.matchIou;
      for (const i of unmatched) {
        if (boxes[i].label !== t.label) continue;
        const v = iou(toRaw(t.box), toRaw(boxes[i]));
        if (v >= bestIou) { bestIou = v; best = i; }
      }
      if (best >= 0) { t.box = boxes[best]; t.seen += 1; t.missed = 0; unmatched.delete(best); }
      else { t.missed += 1; }
    }
    // An unconfirmed track that misses even one frame is thrown away: "3 consecutive frames" means consecutive.
    tracks = tracks.filter((t) => t.confirmed || t.missed === 0);
    for (const i of unmatched) {
      tracks.push({ id: nextId++, label: boxes[i].label, box: boxes[i], seen: 1, missed: 0, confirmed: false, spokenAt: null, spokenBucket: null, spokenZone: null });
    }
    const dropped = tracks.filter((t) => t.missed > cfg.dropAfterFrames);
    if (dropped.some((t) => t.spokenZone === "ahead" && t.spokenBucket === "close")) pendingClear = true;
    tracks = tracks.filter((t) => t.missed <= cfg.dropAfterFrames);
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
      const zone = zoneOf(t.box.cx), bucket = bucketOf(t.box.h, cfg);
      const isNew = t.spokenAt === null;
      const closer = t.spokenBucket !== null && BUCKET_RANK[bucket] > BUCKET_RANK[t.spokenBucket];
      const inPath = zone === "ahead" && t.spokenZone !== null && t.spokenZone !== "ahead";
      if (!isNew && !closer && !inPath) continue;
      if (!isNew && !closer && t.spokenAt !== null && now - t.spokenAt < cfg.repeatSameObjectMs) continue;
      const priority = priorityOf(t, zone, bucket, closer);
      out.push({ track: t, zone, bucket, phrase: { text: phraseFor(t.label, zone, bucket), priority, interrupt: priority === 1 } });
    }
    return out;
  }

  return {
    update(boxes: Box[], nowMs: number): Phrase | null {
      match(boxes);
      const cands = candidates(nowMs).sort((a, b) => a.phrase.priority - b.phrase.priority);
      const gapOk = nowMs - lastPhraseAt >= cfg.minGapMs;
      const pick = cands[0];
      if (pick && (gapOk || pick.phrase.priority === 1)) {
        pick.track.spokenAt = nowMs; pick.track.spokenBucket = pick.bucket; pick.track.spokenZone = pick.zone;
        lastPhraseAt = nowMs;
        pendingClear = false;
        return pick.phrase;
      }
      if (pendingClear && gapOk) {
        const stillBlocked = tracks.some((t) => t.confirmed && t.missed === 0 && zoneOf(t.box.cx) === "ahead" && bucketOf(t.box.h, cfg) === "close");
        pendingClear = false;
        if (!stillBlocked) { lastPhraseAt = nowMs; return { text: "path clear", priority: 4, interrupt: false }; }
      }
      return null;
    },
    reset() { tracks = []; lastPhraseAt = -Infinity; pendingClear = false; },
  };
}
