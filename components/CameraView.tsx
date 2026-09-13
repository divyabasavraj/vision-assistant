"use client";
/**
 * Purpose : Shows the live camera picture full-screen, draws detection boxes over it, and reports
 *           double-taps (pause/resume narration).
 * Depends : a FrameSource to attach the <video> to; Box[] from lib/detector.ts.
 * Question you should be able to answer: why must the <video> be muted and playsInline on phones?
 */
import { useEffect, useRef } from "react";
import type { FrameSource } from "@/lib/frameSource";
import type { Box } from "@/lib/detector";

export function CameraView({ frames, boxes, onDoubleTap }: { frames: FrameSource; boxes: Box[]; onDoubleTap: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastTap = useRef(0);

  useEffect(() => { if (videoRef.current) frames.attach(videoRef.current); }, [frames]);

  useEffect(() => {
    const c = canvasRef.current, v = videoRef.current;
    if (!c || !v) return;
    // The video is object-cover, so map frame fractions onto the displayed (cropped) area.
    c.width = c.clientWidth; c.height = c.clientHeight;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    const vw = v.videoWidth || 1, vh = v.videoHeight || 1;
    const scale = Math.max(c.width / vw, c.height / vh);
    const dx = (c.width - vw * scale) / 2, dy = (c.height - vh * scale) / 2;
    ctx.lineWidth = 3; ctx.font = "bold 18px sans-serif";
    for (const b of boxes) {
      const x = dx + (b.cx - b.w / 2) * vw * scale, y = dy + (b.cy - b.h / 2) * vh * scale;
      const w = b.w * vw * scale, h = b.h * vh * scale;
      ctx.strokeStyle = "#00ff88"; ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "#00ff88"; ctx.fillText(`${b.label} ${(b.score * 100) | 0}%`, x + 4, y + 20);
    }
  }, [boxes]);

  const handleTap = () => {
    const now = Date.now();
    if (now - lastTap.current < 350) onDoubleTap();
    lastTap.current = now;
  };

  return (
    <div className="absolute inset-0 bg-black" onPointerDown={handleTap} role="button" aria-label="Camera view. Double tap to pause or resume narration.">
      <video ref={videoRef} className="h-full w-full object-cover" muted playsInline />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true" />
    </div>
  );
}
