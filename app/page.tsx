"use client";
/**
 * Purpose : The one screen. Start button → Assistant.start(); then camera, captions, presets, hold-to-talk.
 * Depends : lib/assistant.ts, components/*.
 * Question you should be able to answer: why does the app need a Start tap before the camera opens?
 */
import { useEffect, useState } from "react";
import { Assistant, type AssistantState } from "@/lib/assistant";
import { CameraView } from "@/components/CameraView";
import { Caption } from "@/components/Caption";
import { TalkButton } from "@/components/TalkButton";

export default function Page() {
  const [state, setState] = useState<AssistantState | null>(null);
  // Lazy init: the constructor touches no browser APIs, so it is safe during SSR;
  // camera/mic/speaker are only opened by start(), which runs after the Start tap.
  const [assistant] = useState(() => new Assistant(setState));
  useEffect(() => () => assistant.stop(), [assistant]);

  if (!state || state.status === "idle" || state.status === "starting") {
    return (
      <main className="flex h-dvh items-center justify-center bg-black p-6">
        <button type="button" onClick={() => void assistant.start()} disabled={state?.status === "starting"}
          className="h-48 w-full rounded-2xl bg-blue-700 text-4xl font-bold text-white disabled:opacity-50" aria-label="Start the assistant">
          {state?.status === "starting" ? "Starting…" : "Start"}
        </button>
      </main>
    );
  }

  return (
    <main className="relative h-dvh overflow-hidden bg-black">
      <CameraView frames={assistant.frames} boxes={state.boxes} onDoubleTap={() => (state.narrationPaused ? assistant.resumeNarration() : assistant.pauseNarration())} />
      <Caption said={state.caption} heard={state.heard} />
      {state.status === "error" && <p role="alert" className="absolute inset-x-0 top-1/3 bg-red-800 p-4 text-xl text-white">{state.error}</p>}
      <div className="absolute inset-x-0 bottom-1/3 flex gap-2 p-2">
        <button type="button" className="flex-1 rounded-xl bg-gray-800/90 py-4 text-xl text-white" onClick={() => void assistant.handleCommand({ kind: "describe" })} aria-label="Describe the scene">Describe</button>
        <button type="button" className="flex-1 rounded-xl bg-gray-800/90 py-4 text-xl text-white" onClick={() => void assistant.handleCommand({ kind: "read" })} aria-label="Read text in view">Read</button>
      </div>
      <TalkButton onQuestion={(q) => void assistant.handleCommand({ kind: "ask", question: q })} />
      <p className="absolute bottom-1 right-2 text-xs text-gray-400">{state.listening ? "🎤 say “vision, …”" : "🎤 off"} · restarts {state.sttRestarts} · {state.backend ?? "…"} {state.fps} fps · Q&A {state.lastAskMs} ms</p>
    </main>
  );
}
