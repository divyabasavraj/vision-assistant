"use client";
/**
 * Purpose : Backup input for the demo: hold the bottom third of the screen, ask, release.
 *           Uses a one-shot (non-continuous) recognizer so it does not fight the always-on listener.
 * Depends : lib/speech.ts createRecognizer.
 * Question you should be able to answer: why is this button not needed when the phone is worn?
 */
import { useRef, useState } from "react";
import { createRecognizer, type Recognizer } from "@/lib/speech";

export function TalkButton({ onQuestion, disabled }: { onQuestion: (q: string) => void; disabled?: boolean }) {
  const [holding, setHolding] = useState(false);
  const rec = useRef<Recognizer | null>(null);
  const text = useRef("");

  const down = () => {
    if (disabled) return;
    text.current = "";
    rec.current = createRecognizer({
      continuous: false,
      onResult: (t) => { text.current = t; },
      onEnd: () => { setHolding(false); if (text.current) onQuestion(text.current); },
      onError: () => setHolding(false),
    });
    if (!rec.current) return;
    navigator.vibrate?.(60);
    setHolding(true);
    rec.current.start();
  };
  const up = () => rec.current?.stop();

  return (
    <button
      type="button"
      aria-label="Hold to ask a question"
      disabled={disabled}
      onPointerDown={down} onPointerUp={up} onPointerCancel={up} onPointerLeave={up}
      className={`absolute inset-x-0 bottom-0 h-1/3 select-none text-2xl font-bold text-white ${holding ? "bg-red-700" : "bg-blue-800/80"} disabled:opacity-40`}
    >
      {holding ? "Listening… release to ask" : "Hold to ask"}
    </button>
  );
}
