/**
 * Purpose : The one server function. POST { question, image } -> { answer } | { error }.
 * Depends : lib/server/askLogic.ts, lib/server/providers.ts. Runs on Vercel Node.js (Fluid Compute).
 * Question you should be able to answer: what happens when both Gemini and Groq fail?
 */
import { NextResponse } from "next/server";
import { validateAskBody, answerWithFallback, createRateLimiter, type Answerer } from "@/lib/server/askLogic";
import { geminiAnswerer, groqAnswerer } from "@/lib/server/providers";

export const maxDuration = 30;
const allow = createRateLimiter(10, 60_000);   // per instance; good enough for a demo, noted in the report

export async function POST(req: Request) {
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";
  if (!allow(ip)) return NextResponse.json({ error: "retry" }, { status: 429 });

  let body: unknown;
  try { body = await req.json(); } catch { return NextResponse.json({ error: "bad json" }, { status: 400 }); }
  const v = validateAskBody(body);
  if (!v.ok) return NextResponse.json({ error: v.message }, { status: 400 });

  // Record which provider answered (for the evaluation log) without changing the fallback logic.
  let provider = "none";
  const named = (name: string, fn: Answerer): Answerer => async (q, img) => { const t = await fn(q, img); provider = name; return t; };
  const result = await answerWithFallback([named("gemini", geminiAnswerer), named("groq", groqAnswerer)], v.data.question, v.data.image);
  return NextResponse.json("answer" in result ? { ...result, provider } : result, { status: "answer" in result ? 200 : 503 });
}
