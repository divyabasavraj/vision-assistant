"use client";
/**
 * Purpose : Shows the live camera picture full-screen and reports double-taps (pause/resume narration).
 * Depends : a FrameSource to attach the <video> to.
 * Question you should be able to answer: why must the <video> be muted and playsInline on phones?
 */
import { useEffect, useRef } from "react";
import type { FrameSource } from "@/lib/frameSource";

export function CameraView({ frames, onDoubleTap }: { frames: FrameSource; onDoubleTap: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const lastTap = useRef(0);

  useEffect(() => { if (videoRef.current) frames.attach(videoRef.current); }, [frames]);

  const handleTap = () => {
    const now = Date.now();
    if (now - lastTap.current < 350) onDoubleTap();
    lastTap.current = now;
  };

  return (
    <div className="absolute inset-0 bg-black" onPointerDown={handleTap} role="button" aria-label="Camera view. Double tap to pause or resume narration.">
      <video ref={videoRef} className="h-full w-full object-cover" muted playsInline />
    </div>
  );
}
