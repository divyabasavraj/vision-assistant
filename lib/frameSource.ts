/**
 * Purpose : "Give me the current camera frame." Everything else (detector, Q&A) talks to this
 *           interface, never to the camera directly, so a wearable camera can replace the phone later.
 * Depends : browser getUserMedia. Client-only.
 * Question you should be able to answer: what would you have to write to use an ESP32-CAM instead?
 */
export type Frame = { image: CanvasImageSource; width: number; height: number };

export interface FrameSource {
  start(): Promise<void>;
  stop(): void;
  attach(video: HTMLVideoElement): void;   // show the live picture on screen
  getFrame(): Frame | null;                // null until the first frame is ready
}

export class CameraFrameSource implements FrameSource {
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    if (this.video) await this.playOn(this.video);
  }

  attach(video: HTMLVideoElement): void {
    this.video = video;
    if (this.stream) void this.playOn(video);
  }

  private async playOn(video: HTMLVideoElement) {
    video.srcObject = this.stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
  }

  getFrame(): Frame | null {
    const v = this.video;
    if (!v || v.readyState < 2 || v.videoWidth === 0) return null;
    return { image: v, width: v.videoWidth, height: v.videoHeight };
  }

  stop(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }
}
