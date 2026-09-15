"""
Should person get a per-class threshold like curb did?

The flat sweep called 0.35 -> 0.50 "marginal" for person (P +0.083, R -0.053). But recall_by_band
showed person's lost recall is almost all tiny far figures the narrator never speaks. This measures
the trade in the narrator's own bands, plus the precision side the band script does not cover, and
one specific complaint from the field: a chair being called a person.

One model pass at conf 0.35, then every threshold is evaluated offline from the same predictions.
Run: training/.venv/bin/python training/scripts/person_threshold.py
"""
import json
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/Users/divyabasavraj/projects all/PROJECT FINAL")
DATA = ROOT / "training/data/merged"
NAMES = ["person","dog","car","bicycle","motorbike","auto","bus","stairs_up","stairs_down","curb","pole","chair","table","door","crosswalk"]
PERSON, CHAIR = 0, 11
NEAR, CLOSE = 0.25, 0.50          # person scale is 1.0
THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55]

def iou(a,b):
    ix=max(0.0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0.0,min(a[3],b[3])-max(a[1],b[1]))
    inter=ix*iy; ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

model = YOLO(str(ROOT/"training/results/n640_v1/best.pt"))
imgs = sorted((DATA/"images/test").iterdir())
per_image = []   # (person_gt: [(box, hfrac)], other_gt: [(cls, box)], person_preds: [(score, box)])
for i,img in enumerate(imgs):
    if i%300==0: print(f"  {i}/{len(imgs)}", flush=True)
    lp = DATA/"labels/test"/(img.stem+".txt")
    if not lp.exists(): continue
    res = model.predict(str(img), conf=0.35, iou=0.45, imgsz=640, verbose=False, device="cpu")[0]
    h,w = res.orig_shape
    pgt, ogt = [], []
    for line in lp.read_text().strip().splitlines():
        p=line.split()
        if len(p)<5: continue
        c=int(p[0]); cx,cy,bw,bh=map(float,p[1:5])
        box=((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h)
        (pgt.append((box,bh)) if c==PERSON else ogt.append((c,box)))
    preds=[(float(x.conf),tuple(map(float,x.xyxy[0]))) for x in res.boxes if int(x.cls)==PERSON]
    per_image.append((pgt,ogt,preds))

def band(hf): return "far" if hf<NEAR else "near" if hf<CLOSE else "close"

rows=[]
for T in THRESHOLDS:
    hit=defaultdict(int); tot=defaultdict(int); tp=fp=0; fp_on=defaultdict(int)
    for pgt,ogt,preds in per_image:
        claimed=set()
        for score,box in sorted(preds,key=lambda p:-p[0]):
            if score<T: continue
            best,bi=-1,0.5
            for j,(g,_) in enumerate(pgt):
                if j in claimed: continue
                v=iou(box,g)
                if v>=bi: best,bi=j,v
            if best>=0: claimed.add(best); tp+=1
            else:
                fp+=1
                # what is under this false person? the best-overlapping non-person label, if any
                oc,ov=None,0.3
                for c,g in ogt:
                    v=iou(box,g)
                    if v>=ov: oc,ov=c,v
                fp_on[NAMES[oc] if oc is not None else "nothing labelled"]+=1
        for j,(_,hf) in enumerate(pgt):
            tot[band(hf)]+=1; hit[band(hf)]+= j in claimed
    P = tp/(tp+fp) if tp+fp else 0
    rows.append(dict(T=T, P=P, tp=tp, fp=fp, far=hit["far"]/tot["far"], near=hit["near"]/tot["near"], close=hit["close"]/tot["close"],
                     n=dict(tot), fp_on=dict(sorted(fp_on.items(), key=lambda kv:-kv[1]))))

print(f"\n{'thr':>5} {'prec':>6} {'TP':>5} {'FP':>5} {'far R':>7} {'near R':>7} {'close R':>8}   false persons sitting on...")
for r in rows:
    on=", ".join(f"{k} {v}" for k,v in list(r['fp_on'].items())[:4])
    print(f"{r['T']:>5.2f} {r['P']:>6.3f} {r['tp']:>5} {r['fp']:>5} {r['far']:>7.3f} {r['near']:>7.3f} {r['close']:>8.3f}   {on}")
print(f"\nperson test instances by band: {rows[0]['n']}")
(ROOT/"training/results/person_threshold.json").write_text(json.dumps(rows,indent=1))
