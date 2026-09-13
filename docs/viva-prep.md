# Viva preparation — questions a professor may ask

Short answers in plain words. Grows every day.

## Q: What is the project in one sentence?
A web app for blind users: a small object detector that I fine-tuned runs inside the phone browser and speaks obstacles in real time; a large cloud vision model answers spoken questions on demand.

## Q: Why is this a deep learning project if you used a pre-trained model?
The core deliverable is a neural network (YOLO11n) that I trained further on my own merged dataset so it learns new classes (stairs, curb, door, pole…). I measure before vs after, run ablations, and deploy it. Fine-tuning is how almost all real-world detectors are built; training from scratch would need millions of images.

## Q: Why two tiers instead of one big model?
Cloud vision models take 2–4 s per call and are rate-limited, useless for walking. A nano model on the phone gives results in under 100 ms and costs nothing. Each model does what it is good at.

## Q: What is mAP50?
For each class, draw the precision–recall curve counting a prediction as correct when its box overlaps the true box by IoU ≥ 0.5; the area under that curve is AP. mAP50 is the mean of AP over classes. mAP50-95 averages over IoU thresholds 0.5…0.95 and is stricter.

## Q: What is IoU?
Intersection over Union: area where two boxes overlap divided by the area they cover together. 1 = identical, 0 = no overlap.

## Q: What was the baseline?
The original COCO yolo11n with no fine-tuning, scored on my untouched test split after translating my labels into COCO's ids. It gets 0.496 mAP50 on the 8 classes COCO knows (person/car/dog/bus/bicycle/motorcycle/chair/table) and exactly 0 on stairs, curb, door, pole, crosswalk, auto because those classes do not exist in COCO — about 0.26 mAP50 averaged over all 15 of my classes.

## Q: Why did you drop images from the augmented dataset?
Roboflow "augmented" exports contain 2–3 rotated/flipped copies of every training image. If I merged those and re-split, a copy of a training image could land in my test set and inflate the score (data leakage). So for augmented datasets I keep one copy per source image before splitting. I did not apply this to un-augmented datasets because they reuse generic file names ("7.jpg") for different photos.

## Q: What is NMS?
The model proposes many overlapping boxes for one object. Non-maximum suppression sorts by score and drops any box that overlaps an already-kept box of the same class by IoU > 0.45, keeping only the best.

## Q: What is WebGPU vs WASM here?
Two ways ONNX Runtime Web can run the network in a browser: WebGPU uses the phone's graphics chip (fast, newer phones); WebAssembly runs on the CPU (works everywhere, slower). We try WebGPU first and fall back.

## Q: Why 640 input and why does frame rate matter?
640 is what we trained at; halving to 320 is ~4× fewer pixels (faster, less accurate) — that is one of my ablations. The narration engine needs several frames per second to confirm an object over 3 frames and react before the user reaches it.

## Q: How do you know your decode/NMS code is right?
I ran the same ONNX file through ultralytics' own predict() and through my TypeScript decodeOutput+nms on one test image with identical thresholds: same 7 boxes, same classes, scores equal to 2 decimals, corners within half a pixel.

## Q: How does the app decide when to speak?
It tracks each object across frames by matching boxes of the same class with IoU ≥ 0.4. An object must be seen 3 frames in a row before it counts (removes flicker). It speaks only on events: new object, got closer (box height crossed a bucket), moved into the middle third, or the path became clear. At most one phrase per 1.5 s; an object ahead and close can interrupt. Same object is not repeated within 5 s unless it got closer.

## Q: How do you estimate distance without a depth sensor?
From box height as a fraction of the frame: < 25% far, 25–50% near, > 50% close. It is a rough proxy (a child and an adult differ) — listed as a limitation.

## Q: Why is the narrator a pure function with unit tests?
Because it is the safety-relevant logic and it must behave the same on every phone. Feeding it fake box lists lets me prove each rule (confirm, rate limit, priority, clear) without a camera.

## Q: What is fine-tuning and why not train from scratch?
Fine-tuning starts from weights already trained on COCO (millions of boxes) and continues training on my ~7k images. Early layers already detect edges/textures/shapes; only the higher layers and the new output head need to adapt. From scratch, 7k images is far too few — the model would overfit and score badly.

## Q: What does epoch, batch, patience mean?
Epoch = one pass over all training images. Batch = images processed together per gradient step (32). Patience = stop if validation mAP has not improved for 10 epochs (early stopping, avoids overfitting and wasted time). My run never triggered it — val mAP was still improving at epoch 50.

## Q: What is ONNX and why export?
Open Neural Network Exchange — a framework-neutral file describing the network graph and weights. The browser cannot run PyTorch, but ONNX Runtime Web can run ONNX using WebGPU or WebAssembly.

## Q: What does the model output look like?
One tensor 1×(4+nc)×8400 = 1×19×8400 for my 15 classes: for each of 8400 candidate positions, a box (cx, cy, w, h in input pixels) plus one score per class. We keep positions whose best class score passes a threshold and remove overlapping duplicates with NMS.

## Q: What did fine-tuning achieve?
Test mAP50 went from ≈0.26 (COCO model, 0 on 7 of my classes) to 0.667 after 50 epochs on a T4 (1.7 h). Door 0.95 and stairs-up 0.92 are the best; bicycle, pole, curb and crosswalk sit around 0.45–0.52 and are the targets for the second run.

## Q: Which model answers questions and why not run it on the phone?
A cloud vision-language model (Gemini 3.6 Flash, Qwen 3.8 on Groq as fallback). Such models are billions of parameters — far too large for a phone browser. They are only needed occasionally, so 3–5 s latency is acceptable there but not for walking.

## Q: How do you keep API keys safe and quota under control?
Keys live only in `.env.local` on my Mac and in Vercel environment variables, and are used inside the server function. The phone never sees them. The server validates every request with zod, rejects images over 1 MB, and rate-limits each IP to 10 questions per minute.

## Q: What does the system prompt do?
It fixes the model's behaviour: one or two short sentences, concrete, read text exactly, admit when the image is too dark or blurry. Short answers matter because they are spoken aloud. On a photo with no text, "Read" correctly answered "There is no text visible in this image."

## Q: Why did the first phone question fail?
Gemini 3 models "think" before answering by default; on a photo that took 12–20 s, longer than my timeouts. Describing a picture does not need reasoning, so I set `thinkingLevel: "minimal"` — answers now take 3.5–4.5 s consistently.

## Q: What is *really* the deep learning in this project?
The detector: YOLO11n, a convolutional neural network (~100 layers, 2.6 M weights) that maps a 640×640 image to boxes and class scores. Everything else — camera, speech, tracking, the API route — is engineering around that network.

What makes it deep-learning work rather than "used a library":
1. **I changed the weights with data.** The COCO model scored exactly 0 on stairs, curb, pole, door, crosswalk and auto. After 50 epochs of gradient descent on my 7,228 merged training images it scores 0.92 on stairs-up and 0.95 on door. The improvement lives in the weights, not in my code.
2. **Transfer learning.** I did not train from scratch (7k images would overfit). I started from weights trained on millions of COCO boxes, swapped the 80-class head for a 15-class head, and kept training with a small learning rate. Early layers (edges, textures) barely move; the late layers and the head adapt.
3. **Dataset engineering.** Class mapping across three datasets, removing augmented duplicates so nothing leaks into the test split, a held-out test set never seen in training, and a baseline to compare against. 0.26 → 0.667 only means something because of those choices.
4. **The field's metric.** mAP50 over IoU-matched boxes, per class, on unseen data, plus a confusion matrix and failure cases (camels → "dog").
5. **Deployment constraints shaped the model.** Nano size, exported to ONNX, 5 fps on a phone CPU. The ablations (320-px input, quantisation, a longer v2) are the accuracy-vs-speed trade-offs a DL engineer studies.

What is **not** my deep learning: Gemini / Qwen answering "describe this". Those are someone else's billion-parameter models called over an API — I use them, I did not train them. That honesty is the point of the two-tier design: a small model I trained for the always-on safety loop, a large model I call for the rare open-ended question.

**One-sentence version:** I fine-tuned a YOLO11n convolutional detector on a merged 9k-image dataset to learn seven navigation classes COCO doesn't have, raising test mAP50 from 0.26 to 0.67, and deployed it in-browser on a phone; the cloud model is an add-on, not the contribution.


## Q: How do you know something is about to hit you, with one camera and no depth sensor?
Time-to-contact. If an object's box height is h and it is growing at rate dh/dt, it reaches the camera in roughly h / (dh/dt) seconds. No absolute distance is needed — only how fast the thing is filling the frame — so it works with a single uncalibrated lens. Below 1.5 s straight ahead, the phrase becomes priority 1 and interrupts. The rate is computed on a smoothed height (EMA, alpha 0.4), because raw per-frame height is noisy enough to produce nonsense rates. Note this warns *earlier* than the old rule, which only fired once something was already ahead and close.

## Q: Box height is a terrible distance cue. What did you do about it?
Scaled it by how tall each class really is. A curb fills almost none of the frame even at your feet; a bus fills it from 30 m away. A `CLASS_SIZE` table (person 1.0, bus 1.8, pole 1.5, curb 0.1, …) multiplies the far/near thresholds per class, so "close" means roughly the same real distance whatever the object. The honest limitation: curb and crosswalk lie on the ground, so their box *height* is governed by perspective more than by distance — the better cue for those is how low they sit in the frame (cy). That is named as future work, not claimed as done.

## Q: Why is direction a tone rather than a word?
Both, actually. The Web Speech API's voice cannot be panned — `speechSynthesis` does not route through an AudioContext you control — so the panning is a short 660 Hz cue played through a `StereoPannerNode` just before the phrase. A blip in the correct ear is understood faster than the word "left". The words are kept as well, because a user without earbuds would otherwise lose direction entirely.

## Q: Why didn't you train a classifier to read traffic light colour? (discussed, not built)
Because the answer is already fixed by the object's physical design: traffic lights are red-on-top, green-on-bottom by law everywhere. Cropping the detected box, splitting it into thirds and finding the brightest one gives the state with no model, no dataset and no extra inference time — which matters when the detector already only reaches 5 fps on the phone. Using structure you know beats learning something you can derive.


## Q: How does the app know where it is safe to go?
It does not model free space directly — it uses the absence of tracks. The frame is already split into left / ahead / right thirds, so when a priority-1 obstacle blocks the middle, each side is checked for any confirmed track at near or close range. A side with none is usable. Far objects are ignored, because something far away does not stop you stepping across; crosswalk and door are ignored too, because they are destinations rather than obstacles. Both sides blocked gives "stop"; both clear picks the emptier side, tie-breaking left so the instruction cannot oscillate between frames.

The deliberate exception is level changes. Stairs and curbs get no direction at all, because steering is not the correct response to a step — the step is still there whichever way you turn, and advising a sideways move next to stairs is more dangerous than saying nothing. That is a case where the safest design is to give less advice, not more.


## Q: Your device listens all the time. What about the privacy of people around the user?
A real weakness, and I built the choice rather than hide it. The trigger-word listener runs continuously and restarts itself, and Safari does not transcribe on the device — it sends the audio to Apple. So in always-listening mode, every conversation near the wearer leaves the phone, and only the wearer ever consented to that. My own code is narrow (nothing is sent to my server unless the words contain "vision", and pause/resume never leave the device), but that does not change where the audio goes.

The app now has a mic mode toggle. "Hold to talk" never starts the continuous listener at all — the microphone only opens while the button is pressed. The cost is real: a blind user then has to find a button instead of simply speaking, which is exactly the accessibility the hands-free trigger word was for. That is the trade-off, and I don't think there is a clean answer without on-device speech recognition, which the browser does not give me.


## Q: Can the app tell you the way is clear?
Only in a very limited sense, and the wording is deliberately hedged. The first version said "path clear" whenever the object that had been blocking the centre stopped being tracked. Testing on the phone showed the obvious flaw: the object also stops being tracked when the camera turns away from it. The obstacle is still physically in front of the user; the phone just isn't looking at it. A monocular camera cannot tell "the obstacle moved" from "I moved".

So the phrase now says "ahead clear" rather than "path clear" — a claim about the middle third of one frame, not about the world — and it is only spoken while at least one confirmed object is still in view, which is weak evidence that the camera is still watching the same scene rather than pointing at a blank wall. This is a real limitation of a single forward camera with no odometry, not something a better detector would fix.
