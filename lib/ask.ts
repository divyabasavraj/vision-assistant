/**
 * Purpose : Phone side of Q&A: shrink the current frame to a small JPEG and POST it with the question.
 * Depends : Frame from lib/frameSource.ts; the /api/ask route.
 * Question you should be able to answer: why shrink to 640 px and quality 0.7 before sending?
 */
import type { Frame } from "./frameSource";

export function frameToJpeg(frame: Frame, maxSide = 640, quality = 0.7): string {
  const scale = Math.min(1, maxSide / Math.max(frame.width, frame.height));
  const c = document.createElement("canvas");
  c.width = Math.round(frame.width * scale);
  c.height = Math.round(frame.height * scale);
  c.getContext("2d")!.drawImage(frame.image, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", quality).split(",")[1];
}

export type AskResult = { answer: string; ms: number; provider?: string } | { error: "retry" | "offline" | "server"; ms: number };

export async function askServer(question: string, imageBase64: string): Promise<AskResult> {
  const t0 = performance.now();
  const ms = () => Math.round(performance.now() - t0);
  if (typeof navigator !== "undefined" && navigator.onLine === false) return { error: "offline", ms: ms() };
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, image: imageBase64 }), signal: AbortSignal.timeout(35_000),   // server may try two providers at 15 s each
    });
    const data = (await res.json()) as { answer?: string; error?: string; provider?: string };
    if (res.ok && data.answer) return { answer: data.answer, ms: ms(), provider: data.provider };
    if (res.status === 429 || res.status === 503) return { error: "retry", ms: ms() };
    return { error: "server", ms: ms() };
  } catch {
    return { error: navigator.onLine === false ? "offline" : "server", ms: ms() };
  }
}
