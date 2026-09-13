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
