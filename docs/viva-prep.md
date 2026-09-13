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
