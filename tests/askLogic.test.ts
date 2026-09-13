import { describe, it, expect } from "vitest";
import { validateAskBody, answerWithFallback, createRateLimiter, MAX_IMAGE_BASE64 } from "../lib/server/askLogic";

describe("validateAskBody", () => {
  it("accepts a question and a base64 image", () => {
    const r = validateAskBody({ question: "what is this", image: "abc" });
    expect(r).toEqual({ ok: true, data: { question: "what is this", image: "abc" } });
  });
  it("rejects missing fields and oversized images", () => {
    expect(validateAskBody({ question: "" }).ok).toBe(false);
    expect(validateAskBody({ question: "hi", image: "x".repeat(MAX_IMAGE_BASE64 + 1) }).ok).toBe(false);
  });
});

describe("answerWithFallback", () => {
  it("uses the first provider when it works", async () => {
    const r = await answerWithFallback([async () => "A", async () => "B"], "q", "img");
    expect(r).toEqual({ answer: "A" });
  });
  it("falls back when the first provider throws", async () => {
    const r = await answerWithFallback([async () => { throw new Error("429"); }, async () => "B"], "q", "img");
    expect(r).toEqual({ answer: "B" });
  });
  it("returns retry when all fail or answer empty", async () => {
    expect(await answerWithFallback([async () => { throw new Error("x"); }], "q", "img")).toEqual({ error: "retry" });
    expect(await answerWithFallback([async () => "   "], "q", "img")).toEqual({ error: "retry" });
  });
});

describe("createRateLimiter", () => {
  it("allows N per window per key, then blocks, then frees after the window", () => {
    const allow = createRateLimiter(3, 60_000);
    expect([allow("a", 0), allow("a", 1), allow("a", 2), allow("a", 3)]).toEqual([true, true, true, false]);
    expect(allow("b", 3)).toBe(true);
    expect(allow("a", 60_001)).toBe(true);
  });
});
