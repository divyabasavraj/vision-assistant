/**
 * Purpose : The two real vision-LLM callers, both through the Vercel AI SDK so they share one shape.
 *           Gemini first (free tier), Groq's Llama vision as fallback. Keys come from env vars only.
 * Depends : ai, @ai-sdk/google (GOOGLE_GENERATIVE_AI_API_KEY), @ai-sdk/groq (GROQ_API_KEY).
 * Question you should be able to answer: why do we not call these models from the phone directly?
 */
import { generateText, type LanguageModel } from "ai";
import { google } from "@ai-sdk/google";
import { groq } from "@ai-sdk/groq";
import { SYSTEM_PROMPT, type Answerer } from "./askLogic";

// If a model id is retired, check https://ai.google.dev/gemini-api/docs/models and https://console.groq.com/docs/models
export const GEMINI_MODEL = "gemini-3.6-flash";   // 2.5-flash is closed to new accounts (Sep 2026)
export const GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct";

function makeAnswerer(model: LanguageModel, providerOptions?: Parameters<typeof generateText>[0]["providerOptions"]): Answerer {
  return async (question, imageBase64) => {
    const { text } = await generateText({
      model,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: [{ type: "text", text: question }, { type: "file", mediaType: "image/jpeg", data: imageBase64 }] }],
      maxOutputTokens: 160,   // prompt asks for 1-2 sentences; 120 clipped mid-sentence in testing
      timeout: 15_000,
      providerOptions,
    });
    return text;
  };
}

// Gemini 3 "thinks" before answering by default — 10-20 s on a photo question. Minimal thinking is
// plenty for "what is this / read this" and brings it back to ~3-5 s.
export const geminiAnswerer: Answerer = makeAnswerer(google(GEMINI_MODEL), { google: { thinkingConfig: { thinkingLevel: "minimal" } } });
export const groqAnswerer: Answerer = makeAnswerer(groq(GROQ_MODEL));
