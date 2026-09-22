# Vision Assistant — project log, end to end

**Purpose of this file:** one place that tells the whole story of the project so far — *what* was built, *why* it was built that way, *how* it works, what changed and when, what broke, and what each number means. Written so you can re-learn any part before the viva without opening the code.

**Period covered:** 2026-09-11 (day 1) → 2026-09-22 (today). 31 commits, 70 passing unit tests, one trained model, one public deployment.

Companion files: `docs/journal/` (daily entries), `docs/viva-prep.md` (Q&A), `docs/superpowers/specs/…design.md` (the spec), `docs/superpowers/plans/…build.md` (the 13-task plan), `training/results/REPORT.md` (dataset + results with figures).

---

## 0. The project in one page

**What it is.** A web app for blind and low-vision users. The phone sits in a chest pouch, camera facing forward. A small object detector runs *inside the phone browser* and speaks obstacles in real time ("person on your left, a few steps away"). When the user says "vision, what's this?", one photo goes to a cloud vision-language model which answers in a sentence.

**Why it counts as deep learning.** The core is a **YOLO11n convolutional neural network that I fine-tuned** on a 9,035-image navigation dataset I assembled from three public sources. Baseline (COCO weights, no fine-tuning) scores 0.264 mAP50 on my test set; my fine-tuned model scores **0.667**. The cloud model is an add-on for things a 15-class detector cannot do (read text, describe scenes).

**Architecture — two tiers.**

```
phone camera ──► Tier 1: fine-tuned YOLO11n in the browser (ONNX Runtime Web)
                    │  boxes: class, position, size, confidence   (~5–10 fps, free, offline)
                    ▼
                narration engine (lib/narrator.ts: tracking, events, priorities)
                    ▼
                text-to-speech  ──► "pole right in front of you, go left"

mic ──► speech-to-text ──► "vision, …"? ──► POST /api/ask ──► Gemini (fallback Groq)  (~3–4 s, needs internet)
                                                  + current frame        ▼
                                                              text-to-speech ──► "A strip of paracetamol 500 mg"
```

**Why two tiers.** Cloud vision models take 2–4 s per call and are rate-limited — fine for an occasional question, useless for walking. A nano detector on the phone gives sub-100 ms results, works without internet, and costs nothing per frame.

**Tech stack.** Next.js 16 + TypeScript (one project holds the page and the server function; one-command Vercel deploy) · ONNX Runtime Web (runs the model on the phone) · Web Speech API (free speech in/out) · Vercel AI SDK (calls Gemini / Groq) · Ultralytics + PyTorch in a Python 3.12 venv (training) · Vitest + pytest (tests).

---

## 1. Timeline at a glance

| Date | Day | What happened | Commits |
|---|---|---|---|
| Sep 11 | 1 | Spec + plan approved. Project skeleton. Dataset merge script. Baseline label script. Camera / speech / trigger-word app skeleton. Roboflow download script. | 6 |
| Sep 13 | 3 | Two more datasets found; class mappings; merged dataset (9,035 imgs). Baseline eval. **Colab fine-tune n640_v1 → 0.667 mAP50.** Detector in browser. Narration engine. First phone run. `/api/ask` with Gemini→Groq. Narration bundle (distance, time-to-contact, stereo cue). Escape direction. Mic privacy toggle. Two field bugs fixed. | 17 |
| Sep 14 | 4 | Narration rewritten as sentences. Per-class thresholds (measured, not guessed). Recall measured in the narrator's own distance bands. | 3 |
| Sep 15 | 5 | Five field-test fixes to approach/escape logic. Person threshold 0.40 (measured). Crowd simulation. Task 9 data analysis (stairs at distance, class sinks). | 6 |
| Sep 16 | 6 | Production `/api/ask` was failing: API keys were never set in Vercel. Fixed and verified live. | 0 (config) |
| Sep 19 | 9 | Learning walkthrough sessions; `questions-log.md` started. | 0 |
| Sep 22 | 12 | Results report (`training/results/REPORT.md`), labelled sample sheets, this log. | pending |

(Days 2, 7, 8, 10, 11 had no project work recorded.)

---

## 2. Day by day

Each entry: **What** · **Why** · **How** · **What broke** · **Learned** · **Numbers**.

### Sep 11 — Day 1: foundations

**What.** Design spec and 13-task plan approved. Git repo, Next.js app (TypeScript + Tailwind), Vitest, Python 3.12 venv with `ultralytics`. Journal template, viva-prep started. Then, ahead of plan: the dataset merge script, the baseline label-remap script, and the first app pieces (camera `FrameSource`, TTS, always-on trigger-word listener, start screen).

**Why these choices.**
- *Next.js* — one project holds the phone page and the `/api/ask` server function and deploys to Vercel with one command.
- *Vitest* — lets me test the narrator and box decoder on the laptop with fake data; no phone or camera needed.
- *Python 3.12 venv via `uv`* — my Mac ships Python 3.14; PyTorch/ultralytics are safest on 3.12. The venv keeps training packages isolated.
- *Nano model (YOLO11n)* — has to run at usable fps inside a phone browser. Bigger = more accurate but too slow for the target.

**How the merge script works** (`training/scripts/merge_datasets.py`). Reads 2–3 Roboflow YOLO-format datasets → translates each one's class names to my 15 target classes using `training/mappings/<name>.json` → drops boxes for unwanted classes → drops images left with no boxes → drops byte-identical duplicates → pools everything → splits 80/10/10 with a fixed seed → writes `data.yaml` and `counts.json`. Tested with tiny fake datasets in a temp folder (`training/tests/test_merge.py`).

**How the baseline label script works** (`make_baseline_labels.py`). To score the *un-tuned* COCO model on *my* test split, my 15 class ids must be translated to COCO's 80 ids (person→0, car→2 …). Classes COCO doesn't have (stairs, curb, door …) are dropped, so the baseline scores 0 on them by definition.

**What broke.** `create-next-app` refuses a folder that already has files (my `docs/`) → scaffolded in a temp folder and copied in. Vitest 5 needs `@types/node ≥ 22`; the scaffold pinned 20 → bumped to 24.

**Learned.** Why all datasets must share ONE class numbering before training: the model's output is a 15-row score matrix, and row 7 must mean "stairs_up" in every image regardless of which dataset it came from.

**Numbers.** Node 24.18, Next 16.3, ultralytics 8.4.147, torch 2.14, onnxruntime-web 1.29.0.

---

### Sep 13 — Day 3: the big day (data → model → app → phone)

#### 3a. Dataset assembled

**What.** Found two more datasets via the Roboflow Universe search API: `lamao/sod-enect` v1 (10k street images: bicycle, motorcycle, pole, curb) and `joydeep-bhattacharya-ghorv/blind-navigation-fukao` v1 (5.2k Indian street images: auto-rickshaw, stairs, door, table). Already had `melissa-xo1mt/visually-impaired-persons` v6 (9.9k, 43 classes). Wrote all three mappings; ran the merge.

**Why sod-enect v1, not v2.** v2 is v1 with rotate/shear/noise augmentation *baked into the files* (24k images). Augmentation belongs in the training loop, not in the stored dataset — otherwise an augmented copy of a training image can land in val/test and inflate the scores (**data leakage**).

**Why sod's `stairs` was left unmapped.** My targets separate stairs_up / stairs_down and I can't tell which one each sod box is. Wrong labels are worse than fewer labels.

**What broke (1) — Fukao only exists augmented.** Every train image has two copies. Added `--augmented fukao` to the merge: keep one copy per Roboflow source stem (`<stem>.rf.<hash>`) before re-splitting. Made it *opt-in* after discovering that vip reuses generic file names (`7_jpg` × 38 different photos) — a global dedupe silently halved my crosswalk labels.

**What broke (2) — baseline scored the wrong labels.** The per-class table showed "airplane 106" and "train 175" — those are my motorbike and bus counts. Ultralytics `resolve()`s the image directory, so my *symlinked* `images/test` folder led it to `merged/labels` (my 15 ids) instead of `baseline/labels` (COCO ids). Fix: symlink each image file, not the directory. Added a test.

**Learned.** Always sanity-check a metrics table against known instance counts — the wrong-label run "worked" and printed plausible numbers. Data leakage can happen before training even starts.

**Numbers.** Merged: **9,035 images** (7,228 / 903 / 904), **23,314 boxes**, 15 classes, smallest class `auto` with 264 train boxes. Baseline COCO yolo11n on my test split: **mAP50 0.496 on the 8 shared classes, 0 on the 7 new → ≈0.264 over all 15.**

#### 3b. First fine-tune (Colab, run `n640_v1`)

**What.** `training/train.ipynb` on a Colab T4: `yolo11n.pt` → 50 epochs, imgsz 640, batch 32, patience 10, seed 0, default augmentation. Exported `best.pt` → `public/models/best.onnx` (opset 17, 1×3×640×640 → 1×19×8400, 10.6 MB).

**Why fine-tune, not train from scratch.** The COCO weights already know edges, textures, people, cars. Fine-tuning keeps that and adds my 7 new classes with 9k images; from scratch would need 100× the data and time.

**What the output shape means.** 8,400 candidate boxes (every cell of three feature-map grids) × 19 rows (4 box coords + 15 class scores). My decoder picks the best class per candidate, drops scores under threshold, then NMS removes overlapping duplicates.

**Numbers.** 1.7 h. Val mAP50 rose 0.321 (epoch 1 — already above the 0.264 all-class baseline) → 0.688 (epoch 50), still creeping up, so slightly under-trained. **Test: mAP50 0.667, mAP50-95 0.462.** Every class > 0.45. Best: door 0.95, stairs_up 0.92, dog 0.86, bus 0.83. Weakest: bicycle 0.45 (only shared class that dropped, 0.48→0.45), pole 0.49, curb 0.50, crosswalk 0.52. Val losses still falling at epoch 50 → not overfitting. First failure image kept: a camel herd called "six dogs".

#### 3c. Detector in the browser (Task 6)

**What.** `lib/detector.ts`: letterbox the frame to 640×640, run ONNX Runtime Web (WebGPU → WASM fallback), decode the 19×8400 output, NMS, map boxes back to frame coords. Detection loop + fps counter in `Assistant`; box overlay canvas in `CameraView`.

**How I know decode+NMS is right.** Ran the same image through ultralytics in Python and through my TypeScript; identical boxes — for both the COCO model and the fine-tuned one.

**Numbers.** 11 unit tests. First phone run (iPhone 13, Safari, LAN HTTPS): whole pipeline works, **wasm backend, 5 fps** (WebGPU did not engage — open question). Detected chair / door / person; a book was called "car" at ~35–40 % (out-of-vocabulary → nearest class).

#### 3d. Narration engine (Task 7)

**What.** `lib/narrator.ts` — pure logic, no browser dependency, so it is unit-tested with fake box lists. Tracking by class + IoU ≥ 0.4; a new object must appear 3 consecutive frames before it exists (kills flicker); dropped after 10 unseen frames. Four events: new / closer / into-path / clear. Priorities 1–4. One phrase per 1.5 s; same object not repeated within 5 s unless it got closer.

**One planned test was wrong, not the code.** It teleported a person from the left third to the middle in one frame (IoU 0) — which by the spec IS a new object. Rewrote the test to drift the box in steps. Kept the strict matcher rather than adding a looser centre-distance fallback that could merge two different people.

**Numbers.** 9 narrator tests; 20 total.

#### 3e. Phone testing setup

**What broke.** Phones only allow camera/mic on HTTPS. Next's `--experimental-https` needs an admin password to install its CA → generated a cert with `mkcert` directly (`certificates/`, git-ignored) and passed `--experimental-https-key/-cert`. Next 16 also blocks its own JS for non-localhost origins → `allowedDevOrigins` in `next.config.ts`.

#### 3f. Voice Q&A (Task 8)

**What.** `app/api/ask/route.ts` + `lib/server/askLogic.ts` (zod validation, image ≤ 1 MB, 10 requests/min per IP, provider fallback order — pure and tested) + `lib/server/providers.ts` (Gemini first, Groq fallback, both via Vercel AI SDK; keys server-side only). Client: `frameToJpeg` + `askServer`, a beep, Describe / Read preset buttons, and "vision, …" trigger.

**Why the models are what they are.** `gemini-2.5-flash` is closed to new accounts → `gemini-3.6-flash` (Google's own suggestion in the error). Gemini 3 "thinking" made the first phone question take 24 s → `thinkingLevel: "minimal"`, now 3.5–4.5 s. Groq's Llama-4 Scout was retired → `qwen/qwen3.8-27b` (accepts images). Proved the fallback by breaking the Gemini key on purpose: Groq answered in 1.0 s.

**Why not call the models from the phone.** The API keys would be visible in the page source. The server route holds them and rate-limits per IP.

**The two-tier argument in one picture.** The camel photo the detector called "six dogs" — Gemini: "A man wearing traditional clothes is leading a camel by a rope… a large herd of camels in the background" (3.7 s).

#### 3g. Narration bundle (six changes beyond the spec)

1. Far objects only spoken if in the middle third (side clutter far away is noise).
2. `SCORE_THRESHOLD` 0.35 → 0.5 by feel (**reversed on Sep 14** once measured — see below).
3. **Per-class distance.** Box height is a bad distance cue because a curb is short and a bus is tall. A `CLASS_SIZE` table of real-world heights relative to a person scales the far/near/close thresholds: a curb counts as close at box height 0.05 while a person needs 0.5.
4. EMA smoothing of box height (alpha 0.4) so one noisy frame cannot flip a distance bucket.
5. **Time-to-contact** as a priority-1 event: if the box is growing fast enough to fill the frame in < 1.5 s, say "car coming at you fast".
6. **Stereo-panned cue tone** before each phrase (`cue()` in `speech.ts`, `ZONE_PAN`): direction arrives in the correct ear before the words.

**Decision recorded: navigation stays local, not global.** Consumer GPS is ~5 m accurate (10–20 m between buildings); "am I on the footpath or in the road" is a 1 m question. The instrument is less precise than the decision. BlindSquare and Microsoft Soundscape reached the same conclusion.

**Numbers.** 20 narrator tests, 37 total.

#### 3h. Escape direction

**What.** When a priority-1 phrase fires, append a way out: "pole, ahead, close, **go left**". A side is blocked if anything near/close sits in it; far objects don't block; crosswalk/door never block (you walk *towards* them). Both blocked → "stop". Deliberately **no** direction for stairs/curb: you can't side-step a step, and steering someone sideways beside stairs is worse than nothing.

**Numbers.** 8 new tests, 45 total. Three older tests changed expectation (the pole case now correctly says "stop": chair left, table right).

#### 3i. Mic privacy toggle

**Problem noticed on the phone.** The wake-word listener runs `continuous: true` forever, and Safari does speech-to-text *in the cloud* — every conversation near the wearer was being uploaded for as long as the app ran. The wearer consented; bystanders did not.

**Fix.** `MicMode` = "always" | "hold". In "hold" the listener never starts; the mic opens only while the talk button is held. Default stays "always" so the demo is unchanged — the toggle shows the trade-off was understood.

#### 3j. Two field bugs, both mine, both now tested

1. Pointing at a door said "door, ahead, close, **go left**" — routing the user around the thing they are walking towards. `escape()` now returns null for guides (door, crosswalk) and level changes.
2. "path clear" fired whenever the blocker left the frame — including when the *phone turned*. An empty frame is not evidence. Now requires at least one confirmed track still in view and says "**ahead clear**", not "path clear" — all we ever knew was the middle third of one frame.

Also: speech was silent all evening while the cue tone played and captions were right → added `speechDiagnostics()` to the status line instead of guessing; `speak()` was swallowing its own `onerror`.

---

### Sep 14 — Day 4: measure instead of guess

#### 4a. Sentences instead of comma-lists (`f327b76`)

**Why.** Field testing called the narration "clumsy". Two causes, one of them a safety bug: "near" and "close" are synonyms in English — heard with no screen at walking speed they carry no distinguishable meaning, yet they are exactly the difference between "be aware" and "you are about to walk into it". And a screen reader renders every comma as a pause, so "car, ahead, close" arrived as three disconnected words.

**Change.**
```
car, ahead, close, go left  →  car right in front of you, go left
person, left, near          →  person on your left, a few steps away
car approaching fast, ahead →  car coming at you fast
ahead clear                 →  ahead is clear
```
Plus `spokenName()`: "stairs_up" read literally is "stairs underscore up"; "auto" means nothing spoken → "auto-rickshaw".

**Method.** Tests first: 20 expectations rewritten and watched fail, then the implementation. 49 tests.

#### 4b. Per-class score thresholds, global back to 0.35 (`a7874c7`)

**Why.** The 0.5 threshold was set by feel. Measured on the 904-image test split, mean F1 peaks at 0.35 (0.661) and falls to 0.642 at 0.5. F1 already flatters the higher bar because it prices a miss and a false alarm equally — for a walking aid a *missed* pole is worse than an imagined one, so the true optimum is at or below 0.35.

**Key insight.** The chatter 0.5 was defending against is already handled *for free* by the narrator: a spurious box flickers, so it never matches a track at IoU 0.4, never survives 3 frames, and is never spoken. A real object at 0.38 sits still and gets through. Filtering by confidence suppressed the noise by also suppressing the signal.

**Change.** `decodeOutput` takes a per-class threshold array. Sweep result (0.35 → 0.50): curb −0.010 recall for +0.142 precision → **raised to 0.5**; pole, crosswalk, auto lose 0.12–0.15 recall → stay at 0.35. auto loses recall *and* precision — the root cause is 264 training instances, a Task 9 problem.

**New script.** `training/scripts/threshold_sweep.py` — predictions collected once at the lowest threshold and filtered afterwards (identical to re-running, ~5× cheaper). Write-up `training/results/threshold_sweep.md`. 54 tests.

#### 4c. Recall in the narrator's own distance bands (`2bf09bc`)

**Why.** Flat test-set recall scores the model on work the product throws away: a far object not straight ahead is never spoken. So measure recall per band the app actually uses.

**Result** (conf 0.35, curb 0.5, IoU 0.5):

| band | recall | n |
|---|---|---|
| far (ignored) | 0.408 | 1,043 |
| near (spoken) | 0.657 | 431 |
| close (urgent) | 0.773 | 797 |

Three things fell out. **Pole is not the problem it looked like** — flat 0.41, but 0.84 at close range. **Curb is the real weakness** — 0.48 close on 85 instances; a curb is a trip hazard exactly when closest. **Person 0.48 flat was a measurement artefact** — 0.82 close, 0.73 near; the flat figure is dominated by tiny distant figures (instances < 0.5 % of the frame score 0.086 and are 29 % of all person labels). Only 3 ground-truth person pairs in the whole test set overlap enough for NMS to be implicated, so crowds are a *size* failure, not a crowding failure.

**Script / write-up.** `recall_by_band.py`, `recall_by_band.md`. Task 9 priority order set: curb, then table (0.58), crosswalk (0.57).

---

### Sep 15 — Day 5: field-test fixes to motion logic

Every item here came from walking around with the phone and hearing something wrong.

| Commit | Field observation | Root cause | Fix |
|---|---|---|---|
| `1b9887b` | Person 1 m away walked off; phone said "coming at you fast". | A portrait phone can't fit a whole person at that range; as they step back *more* of them enters the frame, the box grows — height-only time-to-contact reads growth as approach. | Track whether each sample touched the top/bottom edge; no rate when clipped. Nothing lost — anything that close is already "right in front of you" via the bucket path. |
| `114d9e8` | Someone stepped to the user's left; phone said "go left". | Tie-break only counted objects per side; anything left of centre sent you left, into them. | Side with more room beside the blocker's own box wins; count and left-on-tie remain fallbacks. |
| `31e7c44` | Nothing said about a person walking towards you at normal pace, or walking away. | 1.5 s TTC only fires for imminent contact. | Each track keeps 1 s of smoothed heights; +15 % = approaching (priority 2), −15 % = receding (priority 4), moving classes only. Rides inside the existing sentence: "person coming closer, a few steps ahead". Close-range wording unchanged so the urgent warning never re-fires for someone leaving. |
| `103358d` | User wants the direction *before* the person is on top of them. | Escape advice only at priority 1. | Priority-2 phrase in the middle third now carries the same escape advice. Off to one side still gets none — "go right" there would push the user towards the road. |
| `1592cff` | Chairs and other upright things called "person". | Person is 32 % of background false positives on a class that is 28 % of training boxes. | **Person threshold 0.40**, measured in bands: 0.50 would drop close-band recall 0.816→0.759 (people right in front); 0.40 keeps close identical, loses 0.022 near, removes 26 of 139 false persons. The chair-as-person case survives every bar to 0.50 → Task 9 hard-negative item. New script `person_threshold.py`. |
| `fe6b4ce` | In a crowd, a receding person again announced "coming at you fast". Solo tests can't reproduce it. | When a receding person is hidden for a frame by someone nearer, its track adopts the nearer person's box (IoU still ≥ 0.4), height jumps once, one-frame TTC reads a lunge. | Growth rate over the last **two** intervals: a swap's single jump is halved, a real approach keeps growing. 200-run crowd simulation, nobody approaching: false warnings 14 → 4. Genuine 1 s TTC with jitter still warns 96/100 (was 100). |

**Design decisions recorded.** Asymmetric announcements are intentional: there is no "moving away fast" and no "person gone" — silence is the default and only what matters is spoken. Per-class thresholds follow one rule: the precision gain must justify the recall loss *in the bands the narrator speaks*.

**Task 9 analysis done the same day** (not yet acted on):
- **Stairs at distance** is the detector's weak spot: stairs are found reliably up close (100 % in crowd sims, median 0.4 s latency) but the training data has zero far-band stairs instances, so distant stairs are simply never seen.
- **Dog and person act as class sinks** — anything furry/upright the model doesn't know (camel, bag, stair-end) falls into the nearest class. Closed-set detector behaviour; the fix is hard negatives and more variety, not a threshold.
- Data collection priority for v2: distant stairs, chair→person hard negatives, floor clutter→dog hard negatives, curb.

**Deployed to production** after each fix (GitHub push → Vercel).

---

### Sep 16 — Day 6: production Q&A was silently broken

**What.** Live `/api/ask` returned errors for both providers. Local `.env.local` had placeholder keys; **Vercel production had zero environment variables**. Every phone question in production had been falling through both providers.

**How found.** Probed the production endpoint directly; both providers returned `TransformError` → keys invalid/missing. Couldn't read Vercel runtime logs (403 on project credentials) so diagnosed from the outside.

**Fix.** Validated a Gemini key, set `GOOGLE_GENERATIVE_AI_API_KEY` and `GROQ_API_KEY` in Vercel production and `.env.local`. Found and fixed a Groq image-format issue along the way (works with properly formed JPEG). Describe / Read verified live on the phone.

**Learned.** A fallback chain hides a total failure: with both providers dead the app still said "please try again" politely. Check the environment, not just the code.

---

### Sep 19 — Day 9: learning sessions

No code. Step-by-step walkthrough of the whole project from fundamentals (`questions-log.md` tracks the questions and a resume bookmark). Covered: full request flow phone→cloud→back, where the trained model actually runs (on the phone, served as a static file by Vercel; Colab only trains it), what Vercel does, the 13-task plan and status, and why YOLO11n is a CNN (~100 conv layers, 2.6 M weights, 640×640 in, boxes + class scores out in one pass; fine-tuning retrains those layers on my data).

**Viva one-liner settled:** "YOLO11n is a ~100-layer CNN with 2.6 M weights, fine-tuned on 9k navigation images and deployed in-browser on the phone; the CNN is the deep-learning contribution, the cloud VLM is an add-on."

---

### Sep 22 — Day 12: presenting the work

**What.** Professor wants to see the dataset, training and test results. Built:
- `training/results/REPORT.md` — dataset → training → test, all figures inline, opens in VS Code Markdown preview (Cmd+Shift+V).
- `training/scripts/make_sample_sheets.py` + `training/results/samples/{train,val,test}_samples.jpg` — 12 random images per split with ground-truth boxes drawn from the label files, so the raw data can be *shown*, and regenerated live with `--seed N` to prove nothing was cherry-picked.
- A shareable web version of the same report (Claude artifact link, private until shared).
- This log.

---

## 3. Where things stand today

### The model

| | |
|---|---|
| Architecture | YOLO11n — ~100 conv layers, 2.6 M params, one small attention block (C2PSA); the "nano" variant |
| Training | 50 epochs · 640 px · batch 32 · Colab T4 · 1.7 h · seed 0 · default augmentation |
| Dataset | 9,035 images / 23,314 boxes / 15 classes; 80/10/10 split; test never touched |
| Test mAP50 | **0.667** (baseline 0.264) · mAP50-95 0.462 |
| Deployed as | `public/models/best.onnx` (10.6 MB, opset 17) + `labels.json` |
| Thresholds | global 0.35; curb 0.5; person 0.40 (all measured, see §2 Sep 14–15) |
| Weights | `training/results/n640_v1/best.pt` (git-ignored; also on Google Drive) |

### The app — what each file does

| File | Purpose | One question you should be able to answer |
|---|---|---|
| `lib/frameSource.ts` | "Give me the next frame" — phone camera today, swappable for a wearable camera later | Why is the camera behind an interface? |
| `lib/detector.ts` | Load ONNX, letterbox, run, decode 19×8400, per-class thresholds, NMS | What is NMS and why per-class? |
| `lib/narrator.ts` | Tracking (IoU ≥ 0.4, 3-frame confirm), events, priorities, rate limit, distance buckets, time-to-contact, approach/recede trend, escape direction | How do you estimate distance with one camera? |
| `lib/speech.ts` | TTS, stereo cue tone, `speechDiagnostics()` | Why a tone before the words? |
| `lib/listener.ts` | Always-on speech-to-text, trigger word "vision" | What about bystander privacy? |
| `lib/voiceCommands.ts` | pause / resume / describe / read handled locally | Which commands never reach the server? |
| `lib/assistant.ts` | Wires everything: state machine, detection loop, fps, `MicMode`, `onBoxes → narrator → speak` | What happens between a frame and a spoken phrase? |
| `lib/ask.ts` | Client: frame → JPEG (≤ 640 px, q 0.7) → `POST /api/ask` | Why shrink the image? |
| `lib/server/askLogic.ts` | zod validation, ≤ 1 MB, 10/min per IP, fallback order — pure, tested | Why treat an empty answer as an error? |
| `lib/server/providers.ts` | Gemini 3.6 Flash (thinking minimal) → Groq Qwen 3.8 27B via AI SDK | Why not call these from the phone? |
| `app/api/ask/route.ts` | The one server function | Where do the API keys live? |
| `components/CameraView.tsx` | Video + box overlay canvas | — |
| `components/TalkButton.tsx` / `Caption.tsx` | Hold-to-talk backup; large-text caption of everything spoken | — |

**Tests:** 70 Vitest tests in `tests/` (narrator, detector, askLogic, voiceCommands) + pytest for merge and baseline scripts. `tsc` and `eslint` clean.

### Training folder

| Path | What |
|---|---|
| `training/scripts/download_roboflow.py` | Fetch datasets (needs `ROBOFLOW_API_KEY`) |
| `training/scripts/merge_datasets.py` + `mappings/*.json` | Build the merged dataset |
| `training/scripts/make_baseline_labels.py` | COCO-id labels for the baseline eval |
| `training/scripts/threshold_sweep.py` / `person_threshold.py` / `recall_by_band.py` | The three measurement scripts behind the thresholds |
| `training/scripts/make_sample_sheets.py` | Labelled contact sheets |
| `training/train.ipynb` | The Colab notebook for `n640_v1` |
| `training/results/` | `baseline_test.md`, `n640_v1.md`, `n640_v1/` (curves, matrices, json, weights), `threshold_sweep.md`, `person_threshold.md`, `recall_by_band.md`, `REPORT.md`, `samples/` |
| `training/data/` | `raw/{vip,sod,fukao}`, `merged/`, `baseline/` (git-ignored) |

### Deployment

Vercel project `vision-assistant` (team *image-and-speech-recognition*). Every push to `main` deploys. Env vars set in production: `GOOGLE_GENERATIVE_AI_API_KEY`, `GROQ_API_KEY`. `vercel ls --prod` lists the live URLs.

---

## 4. Every number and where it comes from

| Number | Value | Source file |
|---|---|---|
| Raw images downloaded | 25,146 (9,919 + 10,000 + 5,227) | `training/data/raw/*/data.yaml` |
| Merged images | 9,035 = 7,228 + 903 + 904 | `training/data/merged/counts.json` |
| Labelled boxes | 23,314 (18,714 / 2,329 / 2,271) | same |
| Baseline mAP50, 8 shared classes | 0.496 | `training/results/baseline_test.md` |
| Baseline mAP50, all 15 | ≈ 0.264 (= 8/15 × 0.496) | same |
| Fine-tuned test mAP50 / mAP50-95 | 0.667 / 0.462 | `training/results/n640_v1/n640_v1_test.json` |
| Best val mAP50 (epoch 50) | 0.688 | `training/results/n640_v1/results.csv` |
| Training time | 1.7 h on T4 | `n640_v1.md` |
| Mean F1 at 0.35 vs 0.50 | 0.661 vs 0.642 | `threshold_sweep.md` |
| Close-band recall: person / curb / pole | 0.816 / 0.482 / 0.842 | `recall_by_band.md` |
| Person 0.40: false persons removed | 26 of 139, close recall unchanged | `person_threshold.md` |
| Crowd sim false "fast" warnings | 14 → 4 per 200 runs | commit `fe6b4ce` |
| Phone fps (iPhone 13, wasm) | 5 | journal 2026-09-13 |
| Q&A latency, Gemini / Groq | 3.5–4.5 s / 1.0 s | journal 2026-09-13 |
| Unit tests | 70 | `npx vitest run` |

---

## 5. Decisions and their reasons (quick reference)

| Decision | Reason |
|---|---|
| Fine-tune YOLO11n, not train from scratch | Keeps COCO knowledge; 9k images is far too few from scratch |
| Nano, not small/medium | Must run in a phone browser at ≥ 5 fps |
| Model on the phone, questions in the cloud | Real-time needs < 100 ms and no network; reading text needs a large model |
| Merge three datasets | No single one had stairs + curb + pole + auto-rickshaw |
| Un-augmented dataset versions | Baked-in augmentation leaks train images into val/test |
| Leave sod `stairs` unmapped | Can't tell up from down; wrong labels worse than none |
| Test split untouched until the end | Otherwise the test number is not an honest estimate |
| Global threshold 0.35, not 0.5 | Measured F1 peak; narrator's 3-frame confirm already kills flicker |
| curb 0.5, person 0.40, rest 0.35 | Per-class sweep in the narrator's bands; only raise where recall cost is ~0 |
| Distance from box height × class size | No depth sensor; a curb and a bus are different heights |
| No escape direction for stairs/curb | You can't side-step a step |
| "ahead is clear", not "path clear" | We only ever saw the middle third of one frame |
| Mic "hold" mode exists, "always" is default | Bystander privacy vs demo simplicity — the toggle shows the trade-off |
| Local obstacle navigation, not GPS turn-by-turn | GPS is 5–20 m; footpath-vs-road is a 1 m question |
| Next.js + Vercel | Page + API in one repo, one-command deploy, free HTTPS (camera needs HTTPS) |

---

## 6. What is left (from the 13-task plan)

- [ ] **Task 9** — second fine-tune (`n640_v2` from `best.pt`, longer) with targeted data: distant stairs, chair→person and clutter→dog hard negatives, curb. Ablations: 320 vs 640, nano vs small, aug on/off, fp32 vs int8. Swap in the best model.
- [ ] **Task 10** — wearable polish: Screen Wake Lock, earbud play/pause via Media Session, ARIA / screen-reader pass, blindfold test with a friend.
- [ ] **Task 11** — evaluation collection: debug log of every phrase and Q&A with timestamps; 5–10 walking clips scored useful/redundant/wrong; 30 Q&A photos; end-to-end latency.
- [ ] **Task 12** — report per spec §7 (nine items), slides, backup demo video.
- [ ] **Task 13** — rehearse on the actual phone and Wi-Fi; re-read viva-prep; be able to explain every `lib/` file.
- Open question: why does WebGPU not engage on iOS Safari (stuck on wasm at 5 fps)?

---

## 7. Glossary (plain words)

- **Fine-tuning** — continue training an already-trained network on your own data so it learns your classes without forgetting what it knew.
- **Epoch** — one full pass over the training set. **Batch** — how many images are processed together before the weights are updated. **Patience** — stop early if validation doesn't improve for N epochs.
- **mAP50** — for each class, plot precision vs recall over all confidence levels, take the area (AP); a box is "correct" if it overlaps the truth by ≥ 50 % (IoU 0.5); average over classes. **mAP50-95** — the same averaged over IoU 0.5…0.95, so it also rewards tight boxes.
- **IoU** — intersection over union: overlap area ÷ combined area of two boxes; 1 = identical.
- **Precision** — of the boxes the model drew, how many were right. **Recall** — of the real objects, how many the model found. A walking aid cares more about recall.
- **NMS** — non-maximum suppression: when several boxes cover the same object, keep the highest-scoring one and drop the rest that overlap it.
- **Confusion matrix** — rows = true class, columns = predicted; the diagonal is correct; the *background* row/column is misses and false alarms.
- **Data leakage** — any way test images (or near-copies of them) end up influencing training. Makes scores look better than reality.
- **ONNX** — a portable file format for a trained network so it can run outside PyTorch (here: in the browser via ONNX Runtime Web). **WebGPU / WASM** — the two ways the browser can run it: GPU (fast) or CPU (slower, always available).
- **Letterbox** — shrink the frame to fit 640×640 and pad the rest grey, so shapes aren't squashed.
- **Time-to-contact** — how fast a box is growing tells you how soon you'll reach it, without knowing the real distance.
- **EMA smoothing** — exponential moving average; each new value only partly moves the running estimate, so one noisy frame can't flip a decision.
- **Closed-set detector** — it only knows its 15 classes; everything else gets the nearest class (camel → dog). The cloud model is the open-set backup.
