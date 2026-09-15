/**
 * Purpose : Decide WHEN to speak and WHAT to say from a stream of boxes. Pure logic, no browser.
 *           Tracks objects across frames (IoU matching), confirms after 3 frames, speaks only on events
 *           (new / closer / farther / in path / motion change / fast approach / clear), one phrase per 1.5 s,
 *           priority-1 may interrupt.
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
  trendWindowMs: 1000,      // motion is judged against the smoothed height about this long ago
  trendMinSpanMs: 600,      // ...but never against a sample younger than this: too little time to mean anything
  trendRatio: 1.15,         // height up by this factor over the window = coming closer; down by it = moving away
};
export type NarratorConfig = typeof NARRATOR_CONFIG;
export type Zone = "left" | "ahead" | "right";
export type Bucket = "far" | "near" | "close";
export type Trend = "approaching" | "receding" | "still";
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
/**
 * How each label should be said out loud. An underscore read literally ("stairs underscore up") is
 * worse than useless, and a bare "auto" means nothing spoken - the dataset's name for an
 * auto-rickshaw is written shorthand, not speech.
 */
const SPOKEN_NAME: Record<string, string> = {
  stairs_up: "stairs going up",
  stairs_down: "stairs going down",
  auto: "auto rickshaw",
  crosswalk: "crossing",
};
export function spokenName(label: string): string {
  return SPOKEN_NAME[label] ?? label.replace(/_/g, " ");
}

/**
 * A short spoken sentence, not a comma-list of tokens. Two reasons it reads this way.
 *
 * First, the buckets used to be spoken as their own names - "near" and "close" - and those are
 * synonyms in English. Heard with no screen, at walking speed, they are indistinguishable, yet they
 * are the difference between "be aware" and "you are about to walk into it". The distance now rides
 * in phrases that cannot be confused: "a few steps away" against "right in front of you".
 *
 * Second, a screen reader renders every comma as a pause, so "car, ahead, close" arrives as three
 * disconnected words. One clause speaks faster than the tokens it replaces, despite being longer.
 *
 * A far stairs/curb is still worth a heads-up, but saying how far adds nothing you can act on.
 *
 * Motion rides inside the same sentence rather than as a phrase of its own, so a walking person costs
 * one utterance, not two: "person coming closer, a few steps ahead". The close-range wording is left
 * alone — it is the urgent one, and "coming closer" adds nothing to "right in front of you".
 */
export function phraseFor(label: string, zone: Zone, bucket: Bucket, trend: Trend = "still"): string {
  const name = spokenName(label);
  if (LEVEL_CHANGE.has(label) && bucket === "far") return `${name} ahead`;
  const motion = trend === "approaching" ? `${name} coming closer` : trend === "receding" ? `${name} moving away` : null;
  if (zone === "ahead") {
    if (bucket === "close") return `${name} right in front of you`;
    const where = bucket === "near" ? "a few steps ahead" : "further ahead";
    return motion ? `${motion}, ${where}` : `${name} ${where}`;
  }
  if (bucket === "close") return `${name} right next to you, on the ${zone}`;
  if (motion) return `${motion}, on your ${zone}`;
  return bucket === "near" ? `${name} on your ${zone}, a few steps away` : `${name} over on your ${zone}`;
}
/** Deliberately different wording so an urgent warning never sounds like routine narration. */
export function approachPhraseFor(label: string): string {
  return `${spokenName(label)} coming at you fast`;
}

type Track = {
  id: number; label: string; box: Box;
  seen: number; missed: number; confirmed: boolean;
  smoothH: number;          // EMA of box height — every distance decision reads this, never box.h
  updatedAt: number;
  hist: Sample[];                                    // smoothed samples over the last trendWindowMs, oldest first; each notes whether the box touched the top/bottom frame edge
  spokenAt: number | null; spokenBucket: Bucket | null; spokenZone: Zone | null; spokenTrend: Trend | null;
  spokenApproachAt: number | null;
};
type Sample = { h: number; t: number; clipped: boolean };
type Candidate = { track: Track; phrase: Phrase; bucket: Bucket; zone: Zone; trend: Trend; approach: boolean };

/**
 * A box that touches the top or bottom of the frame has a height set by the frame, not by the distance.
 * Someone a metre away does not fit in a portrait phone; as they step back, more of them comes into
 * view and the box GROWS while they move away. Growth from such a sample must never mean approach.
 */
export const EDGE_EPS = 0.01;
export function isClippedVertically(b: Box): boolean {
  return b.cy - b.h / 2 <= EDGE_EPS || b.cy + b.h / 2 >= 1 - EDGE_EPS;
}

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
        t.smoothH = cfg.smoothAlpha * boxes[best].h + (1 - cfg.smoothAlpha) * t.smoothH;
        t.hist.push({ h: t.smoothH, t: now, clipped: isClippedVertically(boxes[best]) });
        while (t.hist.length > 1 && t.hist[1].t <= now - cfg.trendWindowMs) t.hist.shift();   // keep one sample at/past the window edge
        t.updatedAt = now;
        t.seen += 1; t.missed = 0;
        unmatched.delete(best);
      } else { t.missed += 1; }
    }
    // An unconfirmed track that misses even one frame is thrown away: "3 consecutive frames" means consecutive.
    tracks = tracks.filter((t) => t.confirmed || t.missed === 0);
    for (const i of unmatched) {
      const b = boxes[i];
      tracks.push({ id: nextId++, label: b.label, box: b, seen: 1, missed: 0, confirmed: false, smoothH: b.h, updatedAt: now, hist: [{ h: b.h, t: now, clipped: isClippedVertically(b) }], spokenAt: null, spokenBucket: null, spokenZone: null, spokenTrend: null, spokenApproachAt: null });
    }
    const dropped = tracks.filter((t) => t.missed > cfg.dropAfterFrames);
    if (dropped.some((t) => t.spokenZone === "ahead" && t.spokenBucket === "close")) pendingClear = true;
    tracks = tracks.filter((t) => t.missed <= cfg.dropAfterFrames);
  }

  /**
   * Monocular time-to-contact: a box growing at rate dh/dt reaches you in about h / (dh/dt) seconds.
   * No depth sensor needed — only how fast the thing is filling the frame. Infinity = not approaching.
   *
   * The rate is taken over the last two intervals, not one. In a crowd a track that loses its box for
   * a frame takes a nearer person's overlapping box instead, and that single jump looks exactly like a
   * lunge; over two intervals the jump is halved while a real approach, which keeps growing, is not.
   * Either end sample touching the frame edge means the height was set by the frame, so there is no rate.
   */
  function timeToContact(t: Track): number {
    const n = t.hist.length;
    if (n < 3) return Infinity;
    const a = t.hist[n - 3], c = t.hist[n - 1];
    if (a.clipped || c.clipped) return Infinity;
    const dt = (c.t - a.t) / 1000;
    if (dt <= 0) return Infinity;
    const rate = (c.h - a.h) / dt;
    return rate > 0 ? c.h / rate : Infinity;
  }

  /**
   * Is it coming or going? The time-to-contact above only fires for something that will hit you in
   * 1.5 s; a person walking towards you at normal pace never trips it, yet "coming closer" is the
   * single most useful thing to know about a person. Compare the smoothed height with the sample about
   * a second ago. Only things that move on their own get a verdict: a pole does not approach, you do.
   * A clipped sample at either end is no evidence — see isClippedVertically.
   */
  function trendOf(t: Track, now: number): Trend {
    if (!MOVING.has(t.label) || t.hist.length < 2) return "still";
    const base = t.hist[0], latest = t.hist[t.hist.length - 1];
    if (now - base.t < cfg.trendMinSpanMs || base.clipped || latest.clipped) return "still";
    const ratio = latest.h / base.h;
    return ratio >= cfg.trendRatio ? "approaching" : ratio <= 1 / cfg.trendRatio ? "receding" : "still";
  }

  const onScreen = () => tracks.filter((t) => t.confirmed && t.missed === 0 && !GUIDES.has(t.label));
  /** A side is unusable if anything near or close sits in it; something far away won't stop you stepping across. */
  const sideBlocked = (zone: Zone) => onScreen().some((t) => zoneOf(t.box.cx) === zone && bucketOf(t.smoothH, cfg, t.label) !== "far");
  const sideCount = (zone: Zone) => onScreen().filter((t) => zoneOf(t.box.cx) === zone).length;

  /**
   * Where to go when the way ahead is blocked. Returns null for level changes (you cannot side-step a
   * step — it is still there, and steering someone sideways beside stairs is worse than not advising)
   * and for guides: a door ahead is where you are trying to go, so routing around it is backwards.
   * With both sides open, the side with more room beside the blocker's own box wins: a person standing
   * left of centre is walked around on the right, not into. Then the emptier side; an exact tie always
   * resolves left so it cannot dither.
   */
  function escape(blocker: Track): string | null {
    if (LEVEL_CHANGE.has(blocker.label) || GUIDES.has(blocker.label)) return null;
    const left = sideBlocked("left"), right = sideBlocked("right");
    if (left && right) return "stop";
    if (left) return "go right";
    if (right) return "go left";
    const roomLeft = blocker.box.cx - blocker.box.w / 2, roomRight = 1 - (blocker.box.cx + blocker.box.w / 2);
    if (roomLeft !== roomRight) return roomLeft > roomRight ? "go left" : "go right";
    return sideCount("right") < sideCount("left") ? "go right" : "go left";
  }

  function priorityOf(t: Track, zone: Zone, bucket: Bucket, closer: boolean, trend: Trend): 1 | 2 | 3 | 4 {
    if (zone === "ahead" && bucket === "close") return 1;
    if (MOVING.has(t.label) && (closer || trend === "approaching")) return 2;
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

      const trend = trendOf(t, now);
      const cooling = t.spokenApproachAt !== null && now - t.spokenApproachAt < cfg.approachRepeatMs;
      if (zone === "ahead" && !cooling && timeToContact(t) < cfg.ttcWarnSec) {
        out.push({ track: t, zone, bucket, trend, approach: true, phrase: { text: approachPhraseFor(t.label), priority: 1, interrupt: true, zone } });
        continue;
      }

      const isNew = t.spokenAt === null;
      const closer = t.spokenBucket !== null && BUCKET_RANK[bucket] > BUCKET_RANK[t.spokenBucket];
      // The way out is worth hearing too, but only for things that move: a pole getting farther is you walking off.
      const farther = MOVING.has(t.label) && t.spokenBucket !== null && BUCKET_RANK[bucket] < BUCKET_RANK[t.spokenBucket];
      const inPath = zone === "ahead" && t.spokenZone !== null && t.spokenZone !== "ahead";
      // A change of motion is an event; coming to a standstill is not — "person stopped" is noise. Nor is
      // motion at close range: that phrase cannot carry it, and re-firing the urgent warning as someone
      // walks off is the opposite of what happened. The band drop that follows says "moving away".
      const moved = !isNew && bucket !== "close" && trend !== "still" && trend !== t.spokenTrend;
      if (!isNew && !closer && !farther && !inPath && !moved) continue;
      if (!isNew && !closer && !farther && !moved && now - (t.spokenAt ?? -Infinity) < cfg.repeatSameObjectMs) continue;
      const priority = priorityOf(t, zone, bucket, closer, trend);
      out.push({ track: t, zone, bucket, trend, approach: false, phrase: { text: phraseFor(t.label, zone, bucket, trend), priority, interrupt: priority === 1, zone } });
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
        // A direction is useful when something is in your path: priority 1 (blocked now), or priority 2
        // in the middle third (a moving thing coming closer — better to hear the way out before it is on
        // top of you). Priority 2 off to one side gets none: it is not in the path, and "go right" there
        // would only push you towards the road.
        if (pick.phrase.priority === 1 || (pick.phrase.priority === 2 && pick.zone === "ahead")) {
          const way = escape(pick.track);
          if (way) pick.phrase.text = `${pick.phrase.text}, ${way}`;
        }
        pick.track.spokenAt = nowMs; pick.track.spokenBucket = pick.bucket; pick.track.spokenZone = pick.zone;
        // A fast-approach warning already said it is coming; do not follow it with "coming closer".
        pick.track.spokenTrend = pick.approach ? "approaching" : pick.trend;
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
        if (!stillBlocked && visible.length > 0) { lastPhraseAt = nowMs; return { text: "ahead is clear", priority: 4, interrupt: false, zone: "ahead" }; }
      }
      return null;
    },
    reset() { tracks = []; lastPhraseAt = -Infinity; pendingClear = false; },
  };
}
