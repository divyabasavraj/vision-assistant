/**
 * Purpose : The testable brain of /api/ask — request validation, provider fallback order, per-key rate limit.
 *           No network calls here; providers are passed in as functions (see providers.ts).
 * Depends : zod.
 * Question you should be able to answer: why does the fallback treat an empty answer the same as an error?
 */
import { z } from "zod";

export const MAX_IMAGE_BASE64 = 1_400_000;   // ≈ 1 MB of JPEG once base64-encoded
export const SYSTEM_PROMPT =
  "You are assisting a blind person through their phone camera. Answer in one or two short sentences. " +
  "Be direct and concrete. If there is text, read it exactly. If the image is too dark or blurry to tell, say so instead of guessing.";

export const AskBody = z.object({
  question: z.string().trim().min(1).max(300),
  image: z.string().min(1).max(MAX_IMAGE_BASE64),
});
export type AskRequest = z.infer<typeof AskBody>;

export function validateAskBody(body: unknown): { ok: true; data: AskRequest } | { ok: false; message: string } {
  const r = AskBody.safeParse(body);
  return r.success ? { ok: true, data: r.data } : { ok: false, message: r.error.issues.map((i) => i.message).join("; ") };
}

export type Answerer = (question: string, imageBase64: string) => Promise<string>;

/** Try each provider in order; the first non-empty answer wins. All failing (or all silent) -> ask the user to retry. */
export async function answerWithFallback(answerers: Answerer[], question: string, imageBase64: string): Promise<{ answer: string } | { error: "retry" }> {
  for (const answer of answerers) {
    try {
      const text = (await answer(question, imageBase64)).trim();
      if (text) return { answer: text };
    } catch (e) {
      console.warn("provider failed:", (e as Error).message);
    }
  }
  return { error: "retry" };
}

/** Sliding window per key (we key by IP). Returns true when the call is allowed. */
export function createRateLimiter(limit: number, windowMs: number): (key: string, now?: number) => boolean {
  const hits = new Map<string, number[]>();
  return (key, now = Date.now()) => {
    const recent = (hits.get(key) ?? []).filter((t) => now - t < windowMs);
    if (recent.length >= limit) { hits.set(key, recent); return false; }
    recent.push(now);
    hits.set(key, recent);
    return true;
  };
}
