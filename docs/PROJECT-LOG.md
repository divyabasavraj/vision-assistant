# Vision Assistant — project log, end to end

**Purpose of this file:** one place that tells the whole story of the project so far — *what* was built, *why* it was built that way, *how* it works, what changed and when, what broke, and what each number means. Written so you can re-learn any part before the viva without opening the code.

**Period covered:** 2026-09-11 (day 1) → 2026-09-30. Three trained detector versions (v1 shipped), 73 passing unit tests, one public deployment. The report required by spec §7 is §9.

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
| Sep 22 | 12 | Results report (`training/results/REPORT.md`), labelled sample sheets, this log. Task 9 scaffolding (v2 decisions, ablation runner, int8). | 2 |
| Sep 30 | 20 | Task 9 run on Kaggle T4: v2 (oversampled) 0.653 < v1 0.667, so **v1 stays shipped**. Four ablations: 640 px, nano and augmentation all justified. Unified `evaluate.py`. §8 evaluation study. v3 (optimizer pinned) 0.670, within noise and lower app recall, so v1 still ships. | 3 |

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

### Sep 30 — Day 20: Task 9 results (v2 and ablations)

**What.** The overnight Mac queue from Sep 22 was too slow, so v2 and the four ablation arms were
moved to Kaggle (free T4 GPU, two kernels in parallel, `training/kaggle/`). v2 gained rare-class
oversampling first (`training/scripts/oversample.py`, repeat-factor sampling, Decision 6 in
`v2_decisions.md`). Both kernels finished in ~1 h. One script, `training/scripts/evaluate.py`,
now produces every evaluation number for any weights file.

**Result.** v2 = 0.653 mAP50, below v1's 0.667. An automatic optimizer choice (SGD in v1 → AdamW in v2) plus
re-run warm-up cost more than 30 epochs won back, and oversampling traded common-class recall for rare-class precision. **v1
stays in `public/models/`**; no app change. Ablations: 320 px −0.067, augmentation off −0.033,
yolo11s +0.041 but 2.3× slower and lower recall at the app thresholds. Full write-up §8.8–8.9.

**Lesson.** A negative result is still a result if you can say why from the curves. When continuing
from trained weights, pin the optimizer, lower the learning rate and skip warm-up; never leave
`optimizer: auto`, because it changes with run length. Change one thing per run.

**v3, same day.** To test that diagnosis, v3 repeated v2 with only the training recipe fixed: SGD
pinned, `lr0 0.001` (a tenth of v1's), no warm-up, no oversampling (`training/kaggle/`, 33 min on
T4). The collapse is gone: val mAP50 was 0.667 after epoch 1 and ended at 0.689. But test mAP50 is
0.670 vs 0.667, a gain within noise, and app recall dropped slightly (0.581 → 0.573), so under the
swap rule (better mAP50 *and* no worse app recall) **v1 stays shipped**. Conclusion: the optimizer
was the cause of v2's drop, and v1 was already close to what this data and this model size can
reach. The next gain has to come from data (distant stairs, poles, curbs), not epochs.

**Task 11, same day (desk version).** Added `lib/eventLog.ts` and a **Log** button, which copies
every phrase and Q&A round-trip, with timestamps, latency and provider, as JSON. Ran 30 held-out
test images through the real `/api/ask`: 28 right, 1 partly, 1 wrong; median 3.5 s. Found that
Gemini's free quota (limits of 5 and 20 requests) sends questions to Groq, and that the fallback costs ~8 s
because rate-limit errors are retried first (§8.12).

**Task 12, same day.** The report is §9 of this log, following spec §7 item by item.
`failure_cases.py` picks the worst mistake of each type by a rule; the most useful finding was
that the top "phantom" is a real car the dataset never labelled, and that three of the six failures are
close-up views unlike the training data.

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
| `lib/eventLog.ts` | Evaluation log: every phrase, question, answer (ms, provider), STT restart; **Log** button copies it as JSON | Why the clipboard, not an upload? |
| `lib/server/askLogic.ts` | zod validation, ≤ 1 MB, 10/min per IP, fallback order — pure, tested | Why treat an empty answer as an error? |
| `lib/server/providers.ts` | Gemini 3.6 Flash (thinking minimal) → Groq Qwen 3.8 27B via AI SDK | Why not call these from the phone? |
| `app/api/ask/route.ts` | The one server function | Where do the API keys live? |
| `components/CameraView.tsx` | Video + box overlay canvas | — |
| `components/TalkButton.tsx` / `Caption.tsx` | Hold-to-talk backup; large-text caption of everything spoken | — |

**Tests:** 73 Vitest tests in `tests/` (narrator, detector, askLogic, voiceCommands, eventLog) + pytest for merge and baseline scripts. `tsc` and `eslint` clean.

### Training folder

| Path | What |
|---|---|
| `training/scripts/download_roboflow.py` | Fetch datasets (needs `ROBOFLOW_API_KEY`) |
| `training/scripts/merge_datasets.py` + `mappings/*.json` | Build the merged dataset |
| `training/scripts/make_baseline_labels.py` | COCO-id labels for the baseline eval |
| `training/scripts/threshold_sweep.py` / `person_threshold.py` / `recall_by_band.py` | The three measurement scripts behind the thresholds |
| `training/scripts/make_sample_sheets.py` | Labelled contact sheets |
| `training/scripts/evaluate.py` | Every evaluation number for one weights file |
| `training/scripts/failure_cases.py` | Rule-picked failure images → `training/results/failures/` |
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
| Q&A desk test: accuracy (30 test images) | 28 right, 1 partly, 1 wrong | `docs/eval/qa_sheet.csv` (§8.12) |
| Q&A desk latency: median / p90 | 3.5 s / 9.4 s (Gemini 3.0 s median; Groq fallback 8.5 s) | same |
| v2 test mAP50 / mAP50-95 | 0.653 / 0.447 (not shipped) | `training/results/n640_v2_eval.json` |
| v2 app recall vs v1 | 0.535 vs 0.581 | same, and `n640_v1_eval.json` |
| v3 test mAP50 / mAP50-95 | 0.670 / 0.467 (not shipped) | `training/results/n640_v3_eval.json` |
| v3 app recall / close-band recall vs v1 | 0.573 / 0.778 vs 0.581 / 0.773 | same |
| Ablation mAP50: control / 320 px / no aug / yolo11s | 0.634 / 0.567 / 0.601 / 0.675 | `training/results/ablation_runs/*_test.json` |
| yolo11s ONNX / CPU latency | 37.9 MB / 47 ms (nano 10.6 MB / ~21 ms) | `training/results/s640_e12_eval.json` |
| Unit tests | 73 | `npx vitest run` |

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

- [x] **Task 9**: v2 trained (0.653, not shipped; v1 stays) and all four ablations run, 2026-09-30 (§8.8–8.9). Targeted data collection (distant stairs, hard negatives, curb) deferred as field work (Decision 4). Follow-up v3 = v1 + 30 epochs, SGD pinned, `lr0 0.001`, no warm-up, no oversampling: 0.670, not shipped (§8.8).
- [~] **Task 10**: *dropped from scope 2026-09-30 to protect Tasks 11–13.* Was: wearable polish: Screen Wake Lock, earbud play/pause via Media Session, ARIA / screen-reader pass, blindfold test with a friend.
- [x] **Task 11** (desk version, 2026-09-30): event log + Log button in the app; 30-image Q&A test through the real `/api/ask` with accuracy, latency and provider (§8.12). Walking clips and phone photos not done (no field work, by scope); the Log button is ready if a walk happens.
- [x] **Task 12** (2026-09-30): report per spec §7 as §9 of this log; 6 rule-picked failure cases (`failure_cases.py`). Slides and demo video not made, by scope.
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

---

## 8. Evaluation — every metric, what it means, and how to explain it

*Study section. Every number here is measured on the **904-image test split** (2,271 labelled
boxes) that training never saw, by `training/scripts/evaluate.py` — one script, one code path, so
v1 and v2 are scored by identical arithmetic. Raw output: `training/results/<model>_eval.json`.
The script reproduces v1's published numbers exactly (0.667 / 0.462 / close recall 0.773), which is
the check that it is computing the same thing as before.*

### 8.1 The building block: when is a box "correct"?

A predicted box is a **true positive (TP)** if it has the **same class** as a labelled box and
overlaps it with **IoU ≥ 0.5** (intersection ÷ union of the two boxes). Each labelled box can be
claimed once — highest-scoring prediction first. What is left over:

- **False positive (FP)** — a box the model drew that matches nothing (a false alarm).
- **False negative (FN)** — a labelled object nobody claimed (a miss).

There is no "true negative" in detection: the number of places where there is correctly *nothing*
is infinite, which is why detectors are scored with precision and recall, never accuracy.

### 8.2 Precision, recall, F1 — worked on curb (v1, at the shipped threshold 0.5)

| | count | |
|---|---:|---|
| TP | 47 | curbs found |
| FP | 22 | "curb" said where there was none |
| FN | 53 | curbs missed |

- **Precision** = TP ÷ (TP + FP) = 47 ÷ 69 = **0.681** — when it says curb, it is right 68 % of the time.
- **Recall** = TP ÷ (TP + FN) = 47 ÷ 100 = **0.470** — it finds 47 % of the curbs.
- **F1** = 2PR ÷ (P + R) = **0.556** — the harmonic mean, which is dragged toward the worse of the two.

**The confidence threshold is the dial between them.** Every box comes with a score 0–1; raising
the bar removes low-score boxes, so FPs fall (precision up) but some real objects are dropped too
(recall down). Which way to turn it is a *product* decision: for a walking aid **a missed pole is
worse than an imagined one**, so recall is favoured — that is why the global bar is 0.35, below
the usual 0.5.

### 8.3 AP and mAP — the threshold-free score

- **AP (average precision)** for one class: slide the threshold from 1 down to 0, plot precision
  against recall, take the **area under that curve**. It scores the model at *every* threshold at
  once, so it does not depend on the bar we picked. (Ultralytics samples the curve at 101 recall
  points, the COCO method.)
- **mAP50** = the mean of the 15 per-class APs, with "correct" meaning IoU ≥ 0.5. v1: **0.667**.
- **mAP50-95** = the same thing averaged over ten IoU bars 0.50, 0.55 … 0.95. A box has to be
  *tight* to count at 0.9. v1: **0.462**.

**Reading the gap between them.** A big gap means the model finds the object but draws a loose box.
stairs_down: AP50 0.765 but AP50-95 0.344 — where a staircase "ends" is ambiguous, so boxes are
rarely tight. door: 0.950 vs 0.840 — a door has crisp edges. For this app the loose box does not
matter much (the narrator only uses box height and position), which is why **mAP50 is the headline
number** and mAP50-95 the secondary one.

**Why mAP and not accuracy?** (viva) Accuracy needs true negatives, which detection does not have;
and mAP does not depend on a threshold choice, so two models can be compared fairly.

### 8.4 Two sets of P/R — why they differ

| | precision | recall | at which threshold |
|---|---:|---:|---|
| Ultralytics `val` (standard) | 0.737 | 0.602 | the one confidence that maximises mean F1 over classes |
| App operating point | 0.737 | 0.581 | what the app actually ships: 0.35, curb 0.5, person 0.4 |

The standard numbers are for comparing with other detectors; the app numbers are what the user
actually experiences. Always say which one you are quoting.

### 8.5 Per-class table (v1)

| class | test boxes | AP50 | AP50-95 | P (app) | R (app) | F1 (app) |
|---|---:|---:|---:|---:|---:|---:|
| door | 53 | **0.950** | 0.840 | 0.925 | 0.925 | 0.925 |
| stairs_up | 26 | **0.924** | 0.706 | 0.880 | 0.846 | 0.863 |
| dog | 85 | 0.860 | 0.711 | 0.802 | 0.765 | 0.783 |
| bus | 175 | 0.830 | 0.630 | 0.794 | 0.771 | 0.783 |
| stairs_down | 57 | 0.765 | 0.344 | 0.811 | 0.754 | 0.782 |
| car | 635 | 0.700 | 0.484 | 0.749 | 0.644 | 0.693 |
| chair | 68 | 0.677 | 0.446 | 0.667 | 0.647 | 0.657 |
| table | 46 | 0.638 | 0.358 | 0.875 | 0.457 | 0.600 |
| motorbike | 106 | 0.597 | 0.364 | 0.720 | 0.509 | 0.597 |
| person | 644 | 0.561 | 0.357 | 0.730 | 0.474 | 0.574 |
| auto | 33 | 0.544 | 0.361 | 0.560 | 0.424 | 0.483 |
| crosswalk | 43 | 0.520 | 0.394 | 0.615 | 0.558 | 0.585 |
| curb | 100 | 0.502 | 0.341 | 0.681 | 0.470 | 0.556 |
| pole | 102 | 0.487 | 0.280 | 0.627 | 0.412 | 0.497 |
| bicycle | 98 | **0.453** | 0.316 | 0.577 | 0.459 | 0.511 |

**How to explain the pattern.** Precision is fairly even (0.56–0.93); **recall is what separates
good classes from weak ones.** The weak classes fail by *missing* objects, not by confusing them
with other classes — the confusion matrix's off-diagonal cells are tiny (largest: auto→bus 0.12),
while the background column (missed entirely) is 0.57 for curb, 0.50 for pole. They are thin
(pole), flat (curb, crosswalk) or small and far (person, bicycle).

**Box count does not predict AP.** door has 408 training boxes and scores 0.95; pole has 1,006 and
scores 0.49. Visual difficulty matters more than quantity — which is why oversampling (§8.8) is
expected to help auto and crosswalk, not pole and curb.

**Error bars.** stairs_up has 26 test boxes: one more miss moves its recall by 0.04. Treat person
(644), car (635), bus (175), pole (102), curb (100) as the load-bearing numbers.

### 8.6 Recall by distance band — the product metric

The narrator sorts each object into far / near / close by box height relative to the frame, scaled
by the class's real size (`bucketOf()` in `lib/narrator.ts`). Far objects that are not straight
ahead are never spoken. So:

| band | boxes | recall (v1, app thresholds) | spoken? |
|---|---:|---:|---|
| far | 1,043 (46 %) | 0.406 | only if straight ahead |
| near | 431 | 0.650 | yes |
| close | 797 | **0.773** | yes, urgent |

**How to say it:** "Flat recall is 0.58, but 46 % of the test boxes are far away and the app
deliberately ignores most of them. On objects close enough to matter, recall is 0.77." Report both;
quoting only one misleads in one direction or the other.

Worst close-band class: **curb 0.482** — the model misses half the curbs right at your feet, and a
curb matters most exactly when it is closest. Best: car 0.947, door 0.913.

### 8.7 Baseline and deployment numbers

| | value | what it shows |
|---|---:|---|
| COCO yolo11n, no fine-tuning, all 15 classes | mAP50 0.264 | it cannot know the 7 new classes at all |
| … on the 8 classes COCO already has | 0.496 | even known classes improved with fine-tuning |
| v1 fine-tuned | **0.667** | +0.40 absolute over baseline |
| parameters | 2.59 M | "nano" |
| ONNX file | 10.6 MB (int8: 3.0 MB) | fits a phone download |
| CPU latency, Mac, 2 threads | 21.9 ms (int8: 27 ms) | the model itself is fast |
| phone, iPhone 13 Safari (WASM) | ~5 fps | the browser runtime is the bottleneck, not the model |

### 8.8 v1 → v2: what changed and what it bought

**What v2 was.** 30 further epochs from `n640_v1/best.pt`, same images, same 640 px, batch 32,
plus **repeat-factor oversampling** (LVIS rule, threshold 0.15): an image containing a rare class is
listed more than once in the training list, so auto (the rarest, 264 boxes) is seen ~2.75× as
often. Trained on a Kaggle T4 in 51 min, 2026-09-30. Reasoning: `v2_decisions.md`. Raw numbers:
`training/results/n640_v2_eval.json`.

**Result: v2 is worse, and v1 stays shipped.**

| | v1 | v2 | change |
|---|---:|---:|---:|
| test mAP50 | **0.667** | 0.653 | −0.014 |
| test mAP50-95 | **0.462** | 0.447 | −0.015 |
| app precision | 0.737 | **0.757** | +0.020 |
| app recall | **0.581** | 0.535 | −0.046 |
| recall far / near / close | **0.406 / 0.650 / 0.773** | 0.353 / 0.599 / 0.739 | all down |
| new-class mAP50 (7 non-COCO classes) | **0.670** | 0.662 | −0.008 |
| ONNX / CPU latency | 10.6 MB / 21.9 ms | 10.6 MB / 20.1 ms | same architecture |

Per class (test AP50):

| class | v1 | v2 | | class | v1 | v2 |
|---|---:|---:|---|---|---:|---:|
| door | 0.950 | 0.979 | | person | 0.561 | 0.525 |
| stairs_down | 0.765 | 0.787 | | dog | 0.860 | 0.824 |
| curb | 0.502 | 0.522 | | bus | 0.830 | 0.797 |
| bicycle | 0.453 | 0.464 | | motorbike | 0.597 | 0.551 |
| table | 0.638 | 0.649 | | crosswalk | 0.520 | 0.492 |
| auto | 0.544 | 0.552 | | **pole** | 0.487 | **0.400** |

**Why it did not work — two causes, both visible in the files.**

1. **The optimizer silently changed.** Both runs set `optimizer: auto`, which lets Ultralytics
   ignore `lr0` and pick for itself by run length: over 10,000 iterations → SGD at 0.01, under →
   AdamW at a computed rate. v1 (50 epochs × 226 batches = 11,300) got **SGD, lr 0.01**; v2 (30 ×
   321 = 9,630) got **AdamW, lr 0.000526** (first lines of `training/results/n640_v2/train.log`).
   So v2 continued v1's weights under a different optimizer, with its memory (momentum/moment
   estimates) empty, plus 3 fresh warm-up epochs, whose bias learning rate starts at 0.1. That
   jolt knocked the model out of where v1 had settled: val mAP50 fell from v1's 0.688 to **0.567
   by epoch 3**, and the remaining 27 epochs were spent climbing back, reaching 0.682 at epoch 30,
   still below where it started (`training/results/n640_v2/results.csv`). Decision 3 said "v1 was
   still improving, so train longer". That was right, but continuing needs the *same* optimizer
   at a *smaller* rate and no warm-up (v3: SGD, `lr0 0.001`, `warmup_epochs 0`), so the extra
   epochs refine the weights instead of retraining them.
2. **Oversampling moves the model's attention, it does not add information.** Showing rare-class
   images more often means common classes are a smaller share of each epoch. The result is the
   expected trade: rare classes became *more precise* (auto precision 0.56 → **0.77**, curb 0.68 →
   0.80), but common classes lost recall (person 0.47 → 0.41, pole 0.41 → 0.31). The big losers,
   pole and person, are frequent classes whose difficulty is visual (thin, far away), which
   oversampling cannot fix.

**Did auto overfit from being repeated 2.75×?** No sign of it. Overfitting would show as auto
doing *better* on training images and *worse* on unseen test images. Its test AP50 went slightly
up (0.544 → 0.552) and its test precision up sharply. Across all classes, val box and class losses fell
steadily to the last epoch (1.166 / 1.159) with no upturn. With 33 test boxes, auto's ±0.01 is
noise either way: oversampling neither hurt nor helped it measurably.

**How to say it in the viva:** "v2 was an honest negative result. I changed two things at once:
more epochs with oversampling, and it lost 1.4 points. The curves show why: an automatic setting switched the
optimizer from SGD to AdamW and re-ran warm-up, which cost more than 30 epochs could win back, and oversampling traded common-class recall
for rare-class precision, which is the wrong trade for a walking aid. So I kept v1. The fix is
to pin the optimizer, fine-tune at a tenth of v1's learning rate with no warm-up, and change one
thing at a time."

### 8.8b v3: the fix for v2, and what it shows

**What v3 was.** v2's recipe with its two faults removed and nothing else changed: 30 epochs from
`n640_v1/best.pt`, **SGD pinned, `lr0 0.001`, `warmup_epochs 0`, no oversampling**. Kaggle T4,
33 min, 2026-09-30. Raw numbers: `training/results/n640_v3_eval.json`, curves in
`training/results/n640_v3/`.

| | v1 | v2 | v3 |
|---|---:|---:|---:|
| test mAP50 | 0.667 | 0.653 | **0.670** |
| test mAP50-95 | 0.462 | 0.447 | **0.467** |
| app precision | 0.737 | **0.757** | 0.739 |
| app recall | **0.581** | 0.535 | 0.573 |
| recall far / near / close | **0.406 / 0.650** / 0.773 | 0.353 / 0.599 / 0.739 | 0.396 / 0.624 / **0.778** |
| val mAP50, epoch 3 → last | — | 0.567 → 0.682 | 0.671 → 0.689 |
| ONNX / CPU latency | 10.6 MB / 21.9 ms | 10.6 MB / 20.1 ms | 10.6 MB / 22.4 ms |

Per class, v3 gains on stairs_down (AP50 0.765 → 0.796), auto (0.544 → 0.585), table (0.638 →
0.666) and door (0.950 → 0.970), and loses a little on person (0.561 → 0.542), pole (0.487 →
0.474) and bus (0.830 → 0.813). Close-band recall for pole rose 0.842 → 0.895 and for stairs_down
0.829 → 0.886, while person near-band recall fell 0.712 → 0.676.

**Reading it.**

1. **The v2 diagnosis was right.** With the same starting weights and the optimizer pinned, val
   mAP never collapsed (v2 lost 0.12 in three epochs; v3 never went below 0.66). So the
   drop in v2 was caused by the optimizer switch and warm-up, not by continuing training itself.
2. **But there was little left to win.** +0.003 test mAP50 on 904 images is within run-to-run
   noise. v1 had essentially converged; extra epochs on the same images at a small learning rate
   only refine what is already there.
3. **Why v3 is not shipped.** The swap rule set before the run was: better test mAP50 *and* no
   worse app recall or close-band recall. v3 passes on mAP50 and close-band, but overall app
   recall is 0.008 lower, mostly from near-band persons, the class the narrator speaks about most.
   A tie on the headline metric with a small loss on the metric the user hears is not a reason to
   change the deployed model.

**How to say it in the viva:** "I tested my explanation for v2 by fixing only the optimizer. The
collapse disappeared, which confirms the cause, but the result tied v1. That tells me the model
had converged on this data, so the next improvement has to come from more data for the hard
classes, not more training."

### 8.9 Ablations: what each design choice was worth

Four arms, each changing **one** thing against a control. All start from COCO weights, train **12
epochs, batch 16, 640 px, seed 0**, and are scored on the same 904 test images. Trained on a Kaggle
T4, 2026-09-30. (12 epochs, not v1's 50, so every arm sits below v1 in absolute terms. The
comparison that matters is each arm against its control.)

| arm | what differs | params | mAP50 | Δ vs control | mAP50-95 | new-class mAP50 | train |
|---|---|---:|---:|---:|---:|---:|---:|
| `n640_e12` *(control)* | — | 2.6 M | 0.634 | — | 0.429 | 0.609 | 15 min |
| `n320_e12` | **A: image 320 px** | 2.6 M | 0.567 | **−0.067** | 0.380 | 0.528 | 10 min |
| `s640_e12` | **B: yolo11s (small)** | 9.4 M | **0.675** | **+0.041** | 0.460 | 0.657 | 22 min |
| `n640_noaug_e12` | **C: augmentation off** | 2.6 M | 0.601 | **−0.033** | 0.389 | 0.629 | 14 min |
| v1 int8 | **D: int8 quantisation** | 2.6 M | — | — | — | — | none |

- **A: image size is the most expensive thing to cut.** Halving the side to 320 px quarters
  the pixels and loses 6.7 points, worst on the new classes (−0.08). The objects the app most needs
  (curbs, poles, distant stairs) are small, and at 320 px they shrink to a handful of pixels.
  **640 is justified.**
- **B: the bigger model wins on mAP, but not where it counts.** yolo11s scores +0.041 over the
  control and, even at 12 epochs, 0.675 against v1's 0.667. The cost: 3.6× the parameters, **37.9
  MB ONNX** (vs 10.6) and **47 ms CPU** (vs ~21 ms, 2.3× slower). The phone already runs nano at ~5
  fps, so small would drop to ~2 fps, too slow to warn about something approaching. And at the
  app's thresholds, small's recall (0.565, close 0.762) is *below* v1's (0.581, close 0.773), so
  its mAP edge does not reach the user. **Nano is justified by the phone, not by accuracy.**
- **C: augmentation is worth 3.3 points overall.** Mosaic, flips, colour and scale jitter make the
  model see each image in many versions, which stops it memorising. One surprise: with augmentation
  off, the new-class mean is *higher* (0.629 vs 0.609). A plausible reading is that mosaic crops
  and rescales thin or flat objects (curb, crosswalk) until they are unrecognisable. That is
  worth one follow-up, not a conclusion, since at 12 epochs the arms have wide error bars.
  **Augmentation on is justified.**
- **D: int8 is 3.5× smaller (3.0 MB) but ~30 % slower on CPU** (27 vs 21 ms). Dynamic
  quantisation converts activations back and forth at every layer, and Apple-silicon fp32 kernels
  are already fast. A 10.6 MB download is acceptable, so **fp32 ships**. (Details: `ablations.md`.)

**Conclusion: the shipped configuration survives every ablation.** YOLO11n at 640 px with default
augmentation in fp32 is `n640_v1`, trained for 50 epochs. The only arm that beats it on mAP
(small) loses on the phone and at the operating point. Raw per-arm results:
`training/results/ablation_runs/*_test.json`; small's full evaluation:
`training/results/s640_e12_eval.json`.

### 8.9b Other detector families: is the weak-class problem the data or the model?

pole, curb, bicycle and crosswalk stayed weak through v1, v2, v3 and every ablation. Two readings
fit that: the data cannot teach these classes any better (a data ceiling), or YOLO11n cannot learn
what the data contains (a model ceiling). To tell them apart, two other detector families were
trained under the §8.9 protocol: same data, 12 epochs, batch 16, 640 px, seed 0, `optimizer: auto`
(AdamW, lr 0.000526, the same as the control) and the same 904 test images. Run on a Kaggle T4 on
2026-10-06.

- **YOLO26n**: the same size class as YOLO11n (2.5 M parameters), with a newer NMS-free head. It
  tests whether a better small design fixes the problem.
- **RT-DETR-l**: a transformer detector (32.8 M parameters). Attention lets each object query
  look at the whole image. It tests whether a much stronger model finds more in the same data.

| run | family | params | GFLOPs | mAP50 | Δ | mAP50-95 | new-class mAP50 | T4 inference | train |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `n640_e12` *(control)* | YOLO11n | 2.6 M | 6.5 | 0.634 | — | 0.429 | 0.609 | 3.8 ms | 15 min |
| `s640_e12` *(§8.9 B)* | YOLO11s | 9.4 M | 21.7 | 0.675 | +0.041 | 0.460 | 0.657 | 9.4 ms | 22 min |
| `y26n640_e12` | YOLO26n | 2.5 M | 5.9 | 0.646 | +0.013 | 0.443 | 0.606 | 3.8 ms | 18 min |
| `rtdetrl640_e12` | RT-DETR-l | 32.8 M | 110 | **0.738** | **+0.104** | **0.526** | **0.730** | 38.2 ms | 91 min |

**Per class, AP50 change against the control** (weak classes in bold):

| class | control | YOLO11s | YOLO26n | RT-DETR-l | RT-DETR-l P / R |
|---|---:|---:|---:|---:|---:|
| **pole** | 0.409 | −0.008 | +0.022 | **+0.179** | 0.45 / **0.65** |
| **curb** | 0.465 | +0.042 | +0.013 | **+0.122** | 0.52 / 0.54 |
| **bicycle** | 0.515 | +0.021 | −0.001 | **+0.098** | 0.65 / 0.59 |
| **crosswalk** | 0.461 | +0.022 | −0.048 | **+0.005** | 0.56 / 0.53 |
| auto | 0.452 | +0.104 | −0.015 | +0.211 | 0.69 / 0.53 |
| stairs_down | 0.702 | +0.105 | −0.028 | +0.162 | 0.94 / 0.84 |
| motorbike | 0.529 | +0.055 | +0.039 | +0.153 | 0.78 / 0.60 |
| stairs_up | 0.846 | +0.049 | +0.015 | +0.141 | 0.91 / 1.00 |
| person | 0.556 | +0.028 | +0.034 | +0.067 | 0.67 / 0.54 |

(The other six classes rise by 0.03–0.10 with RT-DETR-l. P/R are at Ultralytics' max-F1 point,
not the app's thresholds.)

**What it shows.**

1. **A new design at the same size does not help.** YOLO26n gains 0.013 overall, which is within
   12-epoch noise, and it moves no weak class by more than 0.05 in either direction. At the nano
   budget the head design is not what limits these classes.
2. **pole, curb and bicycle have more in the data than nano gets out.** RT-DETR-l lifts pole by
   0.18, curb by 0.12 and bicycle by 0.10, from the same images and labels in the same 12 epochs.
   For these three classes the data is not the ceiling. The gain is mostly **recall**: pole recall
   0.65 against YOLO26n's 0.38. That fits §8.5, where the weak classes failed by missing objects.
3. **crosswalk does not move with any model** (+0.005 with a 12.7× bigger transformer, −0.048
   with YOLO26n). That makes crosswalk the one weak class that looks like a **data** problem. The
   likely causes are its labels (three source datasets, with differing ideas of where a crosswalk
   ends) and its size (43 test boxes).
4. **The gain cannot be pinned on attention alone.** RT-DETR-l differs from the control in both
   family and size (12.7× the parameters, 17× the compute). YOLO11s (3.6×) barely moves pole
   (−0.008) or bicycle (+0.021), which hints that the transformer's whole-image view matters for
   thin objects. But no arm here is a large CNN or a small transformer, so that is a hypothesis,
   not a finding.
5. **12 epochs probably understates RT-DETR-l.** Transformer detectors usually converge more
   slowly than CNNs. Its +0.104 is likely a lower bound.

**Why RT-DETR-l does not ship.** It needs 110 GFLOPs per frame against nano's 6.5 and takes 38 ms
on a T4 GPU against 3.8 ms. As fp32 ONNX it would be about 128 MB (not exported). The phone runs
nano at ~5 fps in WASM. Scaling by compute puts RT-DETR-l **below 0.5 fps** (an estimate, not
measured), far too slow to warn about something approaching. **v1 stays shipped.**

**What it changes.** §9.10 said the next gain must come from data. That holds for crosswalk. For
pole, curb and bicycle, this run shows the current data supports a much better detector than the
phone can run. The way to use that is **knowledge distillation**: train nano to copy RT-DETR-l's
outputs, or use RT-DETR-l to pseudo-label unlabelled street images for nano. Both are future work.

Raw results: `training/results/ablation_runs/{y26n640_e12,rtdetrl640_e12}_test.json` and
`*_results.csv`. Job: `training/kaggle/push.sh arch`. Weights stay local
(`training/runs/kaggle_arch/`, gitignored).

### 8.10 Honest limitations to volunteer before you are asked

1. **The shipped thresholds were chosen on the test split** (`threshold_sweep.py`,
   `person_threshold.py` read test data). That makes the *app operating point* numbers (§8.4–8.6)
   slightly optimistic. **mAP is unaffected** — it is threshold-free. The clean fix would have been
   to tune thresholds on val; say so.
2. **Labels come from three merged public datasets** with different habits (e.g. whether a bicycle
   rider is also a "person"). Some "false positives" are unlabelled real objects, so true
   precision is somewhat higher than measured.
3. **Small classes have wide error bars** (stairs_up 26, auto 33 test boxes).
4. **Closed set** — anything outside the 15 classes gets the nearest one (camel → dog). The cloud
   Q&A tier is the open-set backup.
5. **No end-user study.** Field walks by the developer, not a blind-user evaluation — future work.

### 8.12 Application evaluation: Q&A accuracy and latency (Task 11, desk version)

**Scope.** The plan asked for walking clips and 30 phone photos. By scope decision (no field work)
this was done at the desk instead, through the *same code path the phone uses*: the real
`/api/ask` route (validation, rate limit, Gemini → Groq fallback), run locally with the production
keys. What is **not** measured: narration precision on real walks (needs clips), and phone-side
costs (camera capture, JPEG encoding, mobile network).

**Method.** 30 test-split images, 2 per class (for pole, the only two that qualified), each one's
dominant object covering ≥ 8 % of the frame, shrunk to 640 px JPEG q 0.7 exactly as `lib/ask.ts`
does. The same question for all 30, the one a blind user would actually ask: *"What is directly in
front of me?"* Verdict: **right** if the answer names the labelled object correctly, **partly** if
it names it but misreads the scene, **wrong** otherwise. One request every 6.5 s to stay under the
route's own 10/min limit. Images: `docs/eval/qa_images/`; every answer: `docs/eval/qa_sheet.csv`.

**Accuracy: 28 right, 1 partly, 1 wrong (93 % right).**

- *Partly* (#1, person): a selfie, answered "you are taking a selfie". The model assumed the
  camera holder is in the photo. A stock-photo artefact; a worn phone camera never sees its user.
- *Wrong* (#7, bicycle): a clean bicycle on white, answered by **Groq** as "a small table with a
  box of tissues". A plain hallucination, on the easiest image in the set. The same class's other
  image, also answered by Groq, was right. One case, so it is a warning, not a measurement: the
  fallback model is less reliable than the primary.
- Beyond the question: #14 read the bus's destination board ("17 LONG BIÊN – NỘI BÀI"), stairs
  answers always gave direction (up/down), and #29–30 named the zebra crossing: the open-set tier
  does what the 15-class detector cannot.

**Latency (server round-trip, laptop on home Wi-Fi):**

| | n | median | mean | p90 | range |
|---|---:|---:|---:|---:|---:|
| all | 30 | 3.5 s | 5.7 s | 9.4 s | 2.5–19.7 s |
| answered by Gemini | 21 | 3.0 s | 3.7 s | 5.2 s | 2.5–9.8 s |
| answered by Groq (after Gemini failed) | 9 | 8.5 s | 10.2 s | 13.0 s | 7.9–19.7 s |

**The finding that matters: Gemini's free tier runs out during normal use.** 9 of 30 questions
fell through to Groq, every one because of Gemini quota errors
(`generate_content_free_tier_requests`; the errors cite limits of 5 and 20, the free tier's short-window and daily caps). So:

1. **The fallback works.** No question went unanswered; the user heard an answer every time.
2. **But falling back is slow: 8.5 s, not Groq's 1.0 s.** When the Gemini key was deliberately
   broken (Sep 13), Groq answered in 1.0 s because an auth error fails at once. A quota error
   (HTTP 429) is *retried* by the AI SDK, 3 attempts with back-off, before our fallback ever
   runs. The Groq rows are mostly time spent waiting on Gemini.
3. **For the demo:** after ~20 questions in a day, every answer comes from Groq, and ~1 in 10
   Groq answers may be wrong. Fix options: set `maxRetries: 0` for Gemini so a quota error falls
   through in under a second, or a paid Gemini key, or both.

**Fix applied (same day): Gemini `maxRetries: 0`** (`lib/server/providers.ts`). Re-tested with 5
images while Gemini's quota was still exhausted: every request now failed Gemini in one attempt
(no more "Failed after 3 attempts") and Groq answered all 5 correctly, including the bicycle it
got wrong before. Round-trips were 1.2 and 2.2 s for two, but 7.4, 15.2 and 15.5 s for the other
three. That time was Groq itself, not Gemini. So the retry cost is gone, but Groq's free tier is
uneven under load. Honest summary: with Gemini available, ~3 s; on fallback, 1–15 s. A paid Gemini
key is the real fix for a demo day.

**How to say it in the viva:** "I tested Q&A on 30 held-out images through the real server route.
28 were right. The one wrong answer came from the fallback model. The test also showed that the
free Gemini quota runs out after about 20 questions a day. The fallback caught every one, but it
took 8 seconds instead of 1, because the SDK retries rate-limit errors before giving up. I turned
those retries off for Gemini; the remaining delay is Groq's own free-tier speed."

### 8.11 Viva questions on evaluation — short answers

- **What is mAP50?** Mean over classes of the area under the precision–recall curve, counting a box
  correct at IoU ≥ 0.5.
- **Why is mAP50-95 lower?** It also demands tight boxes, up to 95 % overlap.
- **Why not accuracy?** No true negatives in detection; mAP is threshold-free.
- **Which matters more here, precision or recall?** Recall — a missed obstacle is worse than a false
  warning. Hence 0.35, not 0.5.
- **Why is person only 0.56 when it has the most data?** 51 % of test persons are tiny far figures;
  close-band person recall is 0.82.
- **Why is your test number honest?** Test split untouched during training and model selection
  (best.pt is chosen on val); the one exception, threshold tuning, is disclosed in §8.10.
- **Why is v2 worse than v1?** `optimizer: auto` switched SGD → AdamW for the shorter run and warm-up re-ran (val mAP fell 0.688 →
  0.567 in 3 epochs, then only recovered to 0.682), and oversampling traded common-class recall for
  rare-class precision. Negative result, kept v1.
- **Did fixing the optimizer help (v3)?** It fixed the collapse (val never went below 0.66) but
  gained only 0.003 test mAP50, within noise, and app recall fell 0.581 → 0.573. v1 had already
  converged; more epochs on the same data cannot add much. Kept v1.
- **Why nano and not small, when small scored higher?** Small is 3.6× bigger and 2.3× slower, which
  means ~2 fps on the phone, and its recall at the app's thresholds is lower anyway.
- **What did the ablations show?** 640 px is worth 6.7 points and augmentation 3.3. int8 is smaller
  but slower on this hardware. The shipped configuration survives every one.
- **Are the weak classes a data or a model problem?** Mostly the model's capacity. RT-DETR-l on
  the same data lifts pole +0.18, curb +0.12, bicycle +0.10; YOLO26n at nano size does nothing.
  crosswalk stays flat with every model, so that one is data. RT-DETR-l is far too slow for the
  phone; distilling it into nano is the future-work route (§8.9b).
- **How accurate is the Q&A?** 28/30 right on held-out images (§8.12); the wrong one came from the
  fallback model. Median 3.5 s, p90 9.4 s; slow cases are all Gemini quota fall-throughs.
- **What does the confusion matrix tell you?** Errors are misses (background column), not class
  confusions — so the fix is recall (more data, longer training, oversampling), not relabelling.

---

## 9. Report (spec §7, the nine items)

*Task 12. The report required by spec §7, item by item, in the spec's order. Each item gives the
result, then where the evidence lives. Most detail is already earlier in this log or in
`training/results/REPORT.md` (the version with the figures inline); this section is the index a
marker reads first. Status is honest: two of the nine were cut to a desk version by the scope
decision of 2026-09-30 (no field work, no video).*

**Problem and system, in one paragraph.** A phone worn on the chest runs a fine-tuned YOLO11n
detector in the browser and speaks nearby obstacles in real time; a spoken "vision, …" sends one
photo to a cloud vision-language model for anything the 15 classes cannot cover (text, scenes).
Two tiers because real-time warnings need < 100 ms with no network, while reading and describing
need a large model. Full picture: §0; design decisions: §5.

### 9.1 Dataset summary — done

9,035 images, 23,314 boxes, 15 classes, merged from three public Roboflow datasets (vip, sod-enect,
Fukao), un-augmented versions only, deduplicated, split 80/10/10 with a fixed seed: **7,228 /
903 / 904** images. Smallest class: auto (264 train boxes). Test split untouched until evaluation.

- Per-class bar chart: `training/results/n640_v1/fig_class_counts.png`
- Sample images with boxes (12 per split, regenerable with `--seed`): `training/results/samples/`
- Merge method, leakage decisions (sod v1 not v2, Fukao dedupe, sod `stairs` unmapped): §2 Sep 13a
- Numbers: §4, `training/data/merged/counts.json`

### 9.2 Main results — done

| | mAP50, all 15 | mAP50, 8 COCO classes | mAP50, 7 new classes |
|---|---:|---:|---:|
| YOLO11n COCO weights, no fine-tuning | 0.264 | 0.496 | 0 (cannot know them) |
| **YOLO11n fine-tuned (v1, shipped)** | **0.667** | 0.665 (7 of 8 up; bicycle 0.48 → 0.45 the exception) | 0.670 |

mAP50-95 0.462. Per-class AP, P, R, F1: §8.5. Recall where the app speaks: 0.773 close, 0.650
near (§8.6). Two follow-up runs did not beat v1 (v2 0.653, v3 0.670 within noise with lower app
recall; §8.8, §8.8b). Evidence: `training/results/n640_v1/n640_v1_test.json`,
`baseline_test.md`, `n640_v1_eval.json`.

### 9.3 Ablations — done (all four the spec lists)

| change (one at a time, 12-epoch arms) | effect | verdict |
|---|---|---|
| image 640 → 320 px | −0.067 mAP50 | keep 640 |
| nano → small | +0.041 mAP50, but 3.6× params, 2.3× CPU latency, lower app recall | keep nano |
| augmentation on → off | −0.033 mAP50 | keep augmentation |
| ONNX fp32 → int8 | 3.5× smaller (3.0 MB), ~30 % slower | keep fp32 |

Detail and caveats: §8.9, `training/results/ablations.md`.

**Beyond the spec: other detector families (§8.9b).** YOLO26n (same size, newer head) is level
with the control (+0.013). RT-DETR-l, a 32.8 M-parameter transformer, gains +0.104, including
pole +0.18, curb +0.12 and bicycle +0.10, but needs 17× the compute (under 0.5 fps on the phone,
estimated). crosswalk does not improve with any model, so it is the one weak class limited by data.

### 9.4 Training curves — done

Loss and mAP per epoch for v1: `training/results/n640_v1/results.png` (shown in `REPORT.md` §2).
Val mAP50 0.321 at epoch 1 → 0.688 at epoch 50, val losses still falling: slightly under-trained,
not overfitting. The v2 and v3 curves (`training/results/n640_v2/`, `n640_v3/`) are the evidence for
the optimizer finding in §8.8–8.8b.

### 9.5 Failure cases — done (6)

Picked by a rule, not by browsing: `training/scripts/failure_cases.py` runs v1 at the app's
thresholds over the test split and takes, for each failure type, the largest missed object or the
most confident false box. Green = label, red = prediction. Images: `training/results/failures/`.

| image | what happened | cause (one line) |
|---|---|---|
| `missed_close_curb.jpg` | curb filling 72 % of the frame, not detected | broken curb with debris, seen from above at a tilt; training curbs are clean edges at a distance |
| `missed_close_stairs.jpg` | stairs filling the frame, not detected | seen straight down, no start or end in view; the model reads stairs from their outline, and here there is only texture |
| `missed_close_person.jpg` | a face filling the frame, not detected | the body is cut off by the frame edge; training persons are mostly whole bodies (the same cut-off problem as the Sep 15 time-to-contact fix) |
| `wrong_class.jpg` | a pickup truck (labelled car) called bus at 0.91 | no truck class: a large vehicle between car and bus falls to whichever is closer |
| `phantom.jpg` | "car" at 0.93 over nothing labelled | **the model is right**: a real silver car the dataset never labelled. Some "false positives" are label errors (§8.10 item 2) |
| `n640_v1/failure_camels_as_dogs.jpg` | camel herd called six dogs | closed set: camel is not a class, so the nearest class wins. The cloud Q&A answered it correctly |

The pattern: three of six are **close-up views unlike the training data**, which matters because a
chest camera sees things close up. That is the argument for collecting data from the wearer's
viewpoint, listed under future work.

### 9.6 Narration precision — partly done (no walking clips)

**Not measured as specified.** The spec asks for 5–10 recorded walks scored useful / redundant /
wrong under two threshold settings; walks were cut by the scope decision. What exists instead:

- **Simulated crowd, 200 runs, nobody approaching:** false "coming at you fast" warnings 14 → 4
  after the two-interval fix; a genuine 1 s approach with jitter still warns 96 / 100 (`fe6b4ce`, §2 Sep 15).
- **Offline threshold comparison** (the two-setting part, on the test split instead of walks):
  0.35 vs 0.50 global, mean F1 0.661 vs 0.642; per class in the narrator's bands (§2 Sep 14b,
  `threshold_sweep.md`, `person_threshold.md`).
- **Field observations:** every wrong phrase heard on informal walks was traced to a cause and fixed with a
  test (§2 Sep 13j, Sep 15 table).
- **Tooling ready:** the Log button (`lib/eventLog.ts`) records every phrase with a timestamp; one
  walk plus the copied log gives the precision table.

### 9.7 Q&A accuracy — done (desk version)

30 held-out test images, 2 per class, through the real `/api/ask`: **28 right, 1 partly, 1 wrong**.
Median 3.5 s, p90 9.4 s; Gemini alone median 3.0 s. The one wrong answer came from the fallback
model. Deviation from spec: the images are the 15 navigation classes, not currency / medicine /
label photos, because photographing those is field work. §8.12, `docs/eval/qa_sheet.csv`.

### 9.8 End-to-end latency — partly done

| stage | value | how measured |
|---|---:|---|
| model inference, laptop CPU | 21.9 ms | `evaluate.py` (onnxruntime, 2 threads) |
| detection on the phone (iPhone 13, WASM) | ~200 ms per frame (5 fps) | on-screen fps, Sep 13 |
| object appears → confirmed | ≈ 600 ms (3 frames at 5 fps) | derived from the 3-frame rule |
| stairs, first seen → spoken (simulation) | median 0.4 s | crowd simulation, Sep 15 |
| Q&A round-trip (question → answer text) | median 3.5 s, p90 9.4 s | desk test, §8.12 |

So a warning reaches the user in well under a second, versus 3–4 s for a cloud answer; that
ratio is why there are two tiers. **Not measured:** the text-to-speech start-up time on the phone,
and a camera-to-speech stopwatch run on the device.

### 9.9 Limitations — done

The spec's list, each with where it is discussed:

1. **Not a safety device.** It adds to a cane or guide dog, never replaces one. Recall close up is
   0.77, so about 1 in 4 close objects is missed (§8.6).
2. **Distance is estimated from box height** × a per-class size table, not measured. Cut-off boxes
   break it (§2 Sep 15, first row).
3. **Weak at night / in poor light.** Not measured; the datasets are mostly daylight.
4. **Q&A needs internet**, and the free Gemini quota runs out after about 20 questions a day (§8.12).
5. **Screen must stay on**, which drains the battery (Wake Lock was Task 10, dropped).
6. **The trigger word can misfire** in noise, and always-on listening sends bystanders' speech to
   Safari's cloud recogniser (hold mode exists for this, §2 Sep 13i).
7. **No user study.** Tested by the developer only; no blind participant, no blindfold test.

Plus the evaluation caveats in §8.10: thresholds were tuned on the test split, the merged labels
are incomplete (see `phantom.jpg`), and small classes have wide error bars.

### 9.10 Conclusion and future work

A 2.6 M-parameter detector fine-tuned on 9k assembled images runs in a phone browser at 5 fps and
raises mAP50 from 0.264 to 0.667 on held-out data. Every design choice survived an ablation. Two
further training runs showed the model has converged on this data. A 12.7× larger transformer trained on the same
data (§8.9b) lifts pole, curb and bicycle by 0.10–0.18, so for those three the data supports more
than nano extracts. The way to use that on the phone is **distillation** from that model into nano.
crosswalk did not improve with any model, so for it and the rest the next gain has to come from
**data**:

- images from a chest-height camera, especially close-up curbs, stairs and cut-off people (§9.5);
- distant stairs (zero far-band stairs in the training set) and hard negatives (chair→person);
- a truck class, and a relabelling pass for missed boxes.

Then product work: the dropped Task 10 items (Wake Lock, earbud button, screen-reader pass, a
blindfold test), a paid Q&A key, measured walks with the Log button, WebGPU on iOS for more fps,
and a real user study with blind participants.

**Not produced, by scope (2026-09-30):** slides and the backup demo video.
