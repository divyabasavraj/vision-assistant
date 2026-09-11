# Vision-to-Speech Assistant for Visually Impaired Users — Design Spec

**Date:** 2026-09-11
**Deadline:** 2026-09-22
**Budget:** $0
**Hardware:** MacBook M5 Air (training), any modern phone (demo)
**Graded as:** Deep Learning project, must be deployed to a public URL

---

## 1. What we are building

A mobile web app that helps a blind or low-vision person understand what is in front of them.

It has two modes:

1. **Continuous narration.** The phone camera is always on. The app speaks short phrases only when something changes or matters: "person, left, close", "stairs down, ahead". Silence is the default.
2. **Voice Q&A.** The user holds a big button, asks a question ("what's this?", "read the sign", "is there an empty chair?"), and the app speaks the answer.

The app runs in a phone browser (Chrome or Safari), deployed on Vercel at a public link. No app store, no install.

**Worn, not held.** The phone sits in a chest or neck pouch with the camera facing forward, and the user wears one earbud. The user never touches or looks at the screen. This gives the same three properties as camera glasses (forward-facing camera, hands-free, audio in the ear) at ~₹0. The camera source is a swappable module so a dedicated wearable camera (e.g. ESP32-CAM) can replace the phone camera later without changing the rest of the app.

### Why this is a Deep Learning project

The core of the project is a **fine-tuned object detector** (YOLO11n) trained by us on a navigation dataset, exported to ONNX, and running **inside the phone browser**. The model's constraints (size, speed, quantization) come directly from the deployment target, which gives every experiment a purpose. The report includes baseline-vs-fine-tuned results, ablations, latency benchmarks, and failure analysis.

### What we are NOT building

- A native iOS/Android app
- On-device Q&A (the vision-LLM for questions runs in the cloud)
- A trained wake-word model (we use a text trigger word from the browser's speech-to-text instead; a real on-device wake-word model is a stretch goal)
- Dedicated glasses hardware (Meta glasses, Pi, ESP32-CAM) — stretch goal only
- Depth sensing or true distance measurement
- Background operation (phone must be unlocked with the page open)
- A safety-certified navigation aid

---

## 2. Architecture: two tiers

```
phone camera ──► Tier 1: fine-tuned YOLO11n (in browser, ~10–20 fps)
                    │  boxes: class, position, size, confidence
                    ▼
                narration engine (our logic) ──► text-to-speech
                                                 "person, left, close"

mic (always listening) ──► speech-to-text ──► trigger word? ──► /api/ask (server) ──► vision-LLM
                                             ▲                       │
                                       current frame                 ▼
                                                              text-to-speech
                                              "A strip of paracetamol 500mg"
```

**Tier 1 (always on, on-device, free):** the detector and narration engine. Works with poor or no internet.

**Tier 2 (on demand, cloud, free tier):** a large vision-language model answers questions the small detector cannot — reading text, identifying currency and medicine, describing scenes.

Why two tiers: cloud vision models take 2–4 seconds per call and are rate-limited. That is fine for an occasional question but useless for walking. A small on-device detector gives sub-100 ms results, so narration is real-time and costs nothing.

---

## 3. Dataset and training (the DL deliverable)

### Classes (target 10–12)

| Group | Classes |
|---|---|
| People / animals | person, dog |
| Vehicles | car, bike/motorbike, auto-rickshaw, bus |
| Level changes | stairs up, stairs down, curb/step |
| Obstacles | pole/post, chair, table |
| Structure | door, crosswalk |

Final list may shrink depending on what the merged dataset actually contains. Fewer classes done well beats many done badly.

### Data source

Two or three public datasets from Roboflow Universe tagged "obstacle detection for visually impaired" or similar. Steps:

1. Download in YOLO format.
2. Keep only the classes above; re-map class IDs so all datasets share one list.
3. Remove duplicates and obviously bad labels.
4. Split 80% train / 10% validation / 10% test. **Test set is not touched until final evaluation.**
5. Target size: 3,000–6,000 images.

Time cap: if merging takes more than 4 hours, use the single best dataset as-is and note this in the report.

### Model

- **Base:** YOLO11n (nano, ~2.6M parameters), pre-trained on COCO.
- **Why nano:** must run at usable frame rate inside a phone browser. Bigger models are more accurate but too slow for the target.
- **Method:** fine-tuning — continue training the pre-trained model on our dataset so it learns the new classes (stairs, curb, door, etc.) while keeping what it already knows (person, car).
- **Tool:** `ultralytics` Python library.
- **Where:** first quick run on the Mac (PyTorch MPS); longer runs on free Google Colab or Kaggle GPU.
- **Training config:** image size 640, ~50 epochs, default augmentations (flip, HSV shift, mosaic), early stopping on validation mAP.

### Export

`best.pt` → `best.onnx` (opset 17, fixed input 1×3×640×640 or 1×3×320×320). Optionally an int8-quantized version for the speed ablation. The ONNX file (~10 MB) is served as a static asset.

### Outputs of this stage

- `training/train.ipynb` — the notebook with all runs, plots, and tables
- `training/data.yaml` — class list and paths
- `public/models/best.onnx` — the deployed model
- Baseline and fine-tuned metrics (see Section 7)

---

## 4. App architecture

### Tech stack

| Piece | Choice | Why |
|---|---|---|
| Framework | Next.js (App Router) + TypeScript | One project for page + server function; one-command Vercel deploy |
| Styling | Tailwind CSS | Fast, no design system to learn |
| In-browser inference | ONNX Runtime Web | Runs our ONNX model on phone GPU (WebGPU) or CPU (WASM) |
| Speech in/out | Web Speech API (browser built-in) | Free, no keys, works on mobile Chrome/Safari |
| Cloud VLM calls | Vercel AI SDK | Few lines to call Gemini/Groq, easy provider switch |
| Hosting | Vercel Hobby plan | Free, public HTTPS URL (camera/mic require HTTPS) |

### What runs on the phone (browser)

- Back camera stream → `<video>` element
- Detector: loads `best.onnx`, runs inference on a downscaled frame every ~50–100 ms, returns boxes
- Narration engine: consumes boxes, emits phrases (Section 5)
- Text-to-speech: speaks phrases; can be interrupted
- Hands-free listening: speech-to-text runs continuously; a question starts when the text contains the trigger word ("vision", e.g. "vision, what's this?") and ends when the user stops talking
- Backup trigger: the play/pause button on wired earbuds (Media Session API) starts a question; hold-to-talk on screen still works for the demo
- Q&A client: sends question + one JPEG frame to `/api/ask`, speaks the response
- Frame source abstraction: `lib/frameSource.ts` exposes `getFrame()`; today implemented by the phone camera, later swappable for a streamed wearable camera

### What runs on the server

One route: `POST /api/ask`

- Input: `{ question: string, image: base64 JPEG }`
- Validates size (reject > 1 MB), rate-limits per IP (e.g. 10/min)
- Calls Gemini (free tier) with a fixed system prompt; on rate-limit or error, retries with Groq (Llama vision, free tier)
- Output: `{ answer: string }` or `{ error: "retry" | "offline" }`
- API keys live only in Vercel environment variables

### Folder layout

```
app/
  page.tsx               main screen
  layout.tsx
  api/ask/route.ts       the one server function
lib/
  frameSource.ts         "give me the next frame" — phone camera today, swappable later
  detector.ts            load ONNX model, run inference, decode boxes
  narrator.ts            tracking + event rules + phrase generation
  speech.ts              TTS and STT helpers (Web Speech API)
  ask.ts                 client helper that calls /api/ask
components/
  CameraView.tsx         video + box overlay
  TalkButton.tsx         big hold-to-talk button (demo/backup; not needed when worn)
  Caption.tsx            large on-screen text of what was said
public/
  models/best.onnx
training/
  train.ipynb
  data.yaml
  data/                  (gitignored)
docs/
  journal/               one entry per day (Section 8)
  viva-prep.md           likely professor questions with short answers
  superpowers/specs/     this file
```

### Wearing it

- Phone in a chest/neck pouch, back camera facing forward, roughly chest height
- One earbud (wired preferred, for the button); other ear free for real-world sound
- Screen timeout set to "never" and brightness low — browsers stop the camera when the phone locks, so the screen must stay on. Battery lasts a few hours; noted as a limitation
- Everything is controlled by voice: "vision, what's this?", "vision, read that", "vision, describe", "vision, pause", "vision, resume"

### UI (for the demo and for low-vision users; not needed when worn)

- Dark full-screen; camera view visible for sighted testers and the professor
- One large hold-to-talk button covering the bottom third of the screen as a backup; vibrates once when listening starts
- Double-tap anywhere on the camera area to pause/resume narration (spoken confirmation: "paused" / "listening")
- Two small preset buttons: **Describe** and **Read**
- Everything spoken is also shown as large high-contrast text
- All controls have ARIA labels; works with VoiceOver / TalkBack

---

## 5. Narration engine

Lives in `lib/narrator.ts`. Pure logic: takes a list of boxes per frame, keeps small memory, returns a phrase or nothing. Has no dependency on the camera, model, or browser, so it can be unit-tested with fake box lists.

### Rule 1 — Tracking

Match each new box to the previous frame's boxes by class and IoU (overlap ≥ 0.4). Matched boxes keep their ID; unmatched new boxes get a new ID after appearing in 3 consecutive frames (filters flicker). Objects not seen for 10 frames are dropped.

Per tracked object we store: id, class, box, first-seen frame, last-spoken time, last-spoken height bucket.

### Rule 2 — Events (the only times we speak)

| Event | Trigger | Example phrase |
|---|---|---|
| New | object confirmed (3 frames) | "person, left" |
| Closer | height bucket increased (far→near, near→close) | "person, ahead, close" |
| In path | object's center moved into the middle third | "pole, ahead" |
| Clear | an object that was "ahead, close" is gone | "path clear" |

### Rule 3 — Priority and rate limit

Priority order:
1. Any object "ahead" and "close"
2. Moving classes (person, dog, vehicles) getting closer
3. Level changes (stairs, curb) anywhere
4. Everything else

- At most one phrase every 1.5 s
- A priority-1 event may interrupt current speech
- Same object is not re-announced within 5 s unless it moved to a closer bucket

### Rule 4 — Phrase format

`<class>, <where>[, <how far>]`

- **where:** box center x in left third → "left", middle → "ahead", right third → "right"
- **how far:** box height as fraction of frame: < 0.25 → "far", 0.25–0.5 → "near", > 0.5 → "close"
- Stairs/curb phrases omit distance if "far" (not yet relevant)

Thresholds are constants at the top of the file so they can be tuned and reported.

---

## 6. Q&A flow

1. Speech-to-text runs continuously. When the recognised text contains the trigger word "vision", the app plays a short beep and pauses narration. (Alternatives: earbud play/pause button, or hold the on-screen button.)
2. Recognised words shown live on screen
3. User stops talking (speech-to-text reports the sentence ended) → one frame captured as JPEG (max 640 px long side, ~quality 0.7). Voice commands "pause" / "resume" are handled locally and never sent to the server
4. `POST /api/ask` with question + image
5. Server prompt (fixed): *"You are assisting a blind person through their phone camera. Answer in one or two short sentences. Be direct and concrete. If there is text, read it exactly. If the image is too dark or blurry to tell, say so instead of guessing."*
6. Answer spoken and shown; narration resumes when speech ends

### Error handling

| Case | Behaviour |
|---|---|
| Provider rate-limited | try second provider; if both fail → speak "please try again in a moment" |
| No network | speak "no connection — questions need internet"; narration keeps running |
| STT heard nothing after trigger | speak "I didn't catch that" |
| STT stops by itself (browsers do this after silence) | restart it automatically; log how often for the report |
| Narration phrase spoken by the app is picked up by the mic | ignore recognised text while the app is speaking |
| Image too dark/blurry | model instructed to say so |
| Server error | speak "something went wrong, try again" — never a silent failure |

### Preset questions

- **Describe** → "Describe what is in front of me in two short sentences."
- **Read** → "Read any text you can see, exactly as written."

---

## 7. Evaluation (report contents)

1. **Dataset summary** — counts per split, per-class bar chart, 6–8 sample images with boxes
2. **Main results table** — YOLO11n COCO (no fine-tune) vs. fine-tuned: mAP50 overall, mAP50 on new classes, per-class AP
3. **Ablations** (one change at a time, as many as time allows, minimum three):
   - image size 320 vs. 640
   - nano vs. small
   - with vs. without augmentation
   - ONNX fp32 vs. int8
4. **Training curves** — loss and mAP vs. epoch
5. **Failure cases** — 4–6 wrong predictions with a one-line cause each
6. **Narration precision** — 5–10 recorded walking clips; count useful vs. redundant/wrong phrases; report for two threshold settings
7. **Q&A accuracy** — 30 photos (currency, medicine, labels, signs, scenes); right / partly / wrong; average latency
8. **End-to-end latency** — camera frame → spoken phrase on phone; compared to Q&A round-trip
9. **Limitations** — not a safety device; distance estimated from box size; weak at night; Q&A needs internet; screen must stay on (battery); trigger word can misfire in noisy places; small informal user test only

---

## 8. Learning and documentation (a requirement, not a nice-to-have)

The student must be able to explain every part of the project in a viva. Therefore:

1. **Explain before code.** Before any file is written, a few plain-language sentences say what it does and why. The student okays it first.
2. **Daily journal** — `docs/journal/YYYY-MM-DD.md`: what was built, why that choice, what broke, what was learned. Plain language.
3. **File headers** — every source file starts with a plain-English comment: purpose, dependencies, and one question the student should be able to answer about it.
4. **`docs/viva-prep.md`** — grows through the project: likely professor questions with short answers (why YOLO, why nano, what is mAP50, why not cloud for everything, what is IoU, what is fine-tuning, etc.).
5. **Student runs the important parts** — training on Colab, phone testing, eval clip/photo collection, and the report in their own words.
6. **Small files** — each unit readable in ~10 minutes; no file over ~200 lines without a reason.

---

## 9. Stretch goals (only if days 9–10 have spare time)

- **On-device wake-word model** — a tiny audio classifier (e.g. TF.js Speech Commands fine-tuned on "vision") replacing the text trigger. Would be a second DL component
- **ESP32-CAM as the camera** (~₹500–800) clipped to glasses, streaming frames to the phone over Wi-Fi via a new `FrameSource` implementation. Proves the swappable-camera design

## 10. Timeline

| Day | Date | Work | Milestone |
|---|---|---|---|
| 1 | Sep 11–12 | Datasets downloaded, merged, cleaned; ultralytics set up; baseline eval | Dataset + baseline numbers |
| 2 | Sep 13 | First fine-tune (Colab); export ONNX; inspect results | `best.onnx` v1 |
| 3 | Sep 14 | Next.js app: camera via `FrameSource`, TTS, continuous STT with trigger word; **deploy to Vercel** | Public link, talking camera |
| 4 | Sep 15 | ONNX in browser, box overlay, phone testing; WebGPU/WASM issues. **Decision point:** if not working by end of day, move inference to a server function | Live detection on phone |
| 5 | Sep 16 | Narration engine + unit tests with fake boxes; live tuning | Continuous narration |
| 6 | Sep 17 | `/api/ask`, Gemini + Groq fallback, errors, preset buttons | End-to-end app |
| 7 | Sep 18 | Second fine-tune with fixes; ablation runs; swap in best model | Final model + ablation table |
| 8 | Sep 19 | Accessibility, earbud button, polish, rate limiting; wear it in a pouch and test blindfolded with a friend | Demo-ready |
| 9 | Sep 20 | Evaluation: clips, 30 Q&A photos, latency, failure images | All report numbers |
| 10 | Sep 21 | Report + slides; record backup demo video | Report + video |
| 11 | Sep 22 | Buffer; rehearse on the actual phone; submit | Done |

### Known risks and fallbacks

| Risk | Fallback |
|---|---|
| ONNX Runtime Web fails on the phone browser | Server-side inference via a Python function on Vercel (slower, still demos) |
| Dataset merge takes too long | Use the single best dataset; note it in the report |
| Free API throttled during demo | Second provider + backup video |
| Training slower than expected on Mac | Use Colab/Kaggle GPU for all runs |

---

## 11. Success criteria

- A public URL where a phone worn in a chest pouch narrates obstacles in real time and answers spoken questions, with no touching the screen
- A fine-tuned model with a clear before/after result and at least three ablations
- A report with all nine evaluation items in Section 7
- A `docs/journal/` with an entry for each working day and a filled `docs/viva-prep.md`
- The student can explain every file and every number without help
