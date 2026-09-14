"""
Recall in the bands the narrator actually speaks in.

bucketOf() in lib/narrator.ts buckets on BOX HEIGHT as a fraction of the frame, scaled by the
class's real-world size: far < 0.25*s, near < 0.50*s, else close. Objects that land in "far" and
are not straight ahead are never spoken at all. So test-set recall counts many instances the
application would deliberately ignore. This splits recall by the same bands the app uses.
"""
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/Users/divyabasavraj/projects all/PROJECT FINAL")
DATA = ROOT / "training/data/merged"
CONF, NEAR, CLOSE = 0.35, 0.25, 0.50
NAMES = ["person","dog","car","bicycle","motorbike","auto","bus","stairs_up","stairs_down","curb","pole","chair","table","door","crosswalk"]
SCALE = {"person":1.0,"dog":0.35,"car":0.9,"bicycle":0.65,"motorbike":0.75,"auto":1.0,"bus":1.8,
         "stairs_up":0.6,"stairs_down":0.6,"curb":0.1,"pole":1.5,"chair":0.55,"table":0.45,"door":1.2,"crosswalk":0.15}
CURB_T = 0.5   # the per-class override we just shipped

def iou(a,b):
    ix=max(0.0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0.0,min(a[3],b[3])-max(a[1],b[1]))
    inter=ix*iy; ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

model = YOLO(str(ROOT/"training/results/n640_v1/best.pt"))
imgs = sorted((DATA/"images/test").iterdir())
band = defaultdict(lambda: defaultdict(lambda: [0,0]))   # cls -> band -> [hit,total]

for i,img in enumerate(imgs):
    if i%300==0: print(f"  {i}/{len(imgs)}", flush=True)
    res = model.predict(str(img), conf=CONF, iou=0.45, imgsz=640, verbose=False, device="cpu")[0]
    h,w = res.orig_shape
    lp = DATA/"labels/test"/(img.stem+".txt")
    if not lp.exists(): continue
    gt=[]
    for line in lp.read_text().strip().splitlines():
        p=line.split()
        if len(p)<5: continue
        c=int(p[0]); cx,cy,bw,bh=map(float,p[1:5])
        gt.append((c,(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h,bh))
    preds = sorted([(int(x.cls),float(x.conf),*map(float,x.xyxy[0])) for x in res.boxes], key=lambda p:-p[1])
    claimed=set()
    for c,score,*box in preds:
        if NAMES[c]=="curb" and score < CURB_T: continue      # honour the shipped per-class bar
        best,bi=-1,0.5
        for j,g in enumerate(gt):
            if g[0]!=c or j in claimed: continue
            v=iou(box,g[1:5])
            if v>=bi: best,bi=j,v
        if best>=0: claimed.add(best)
    for j,g in enumerate(gt):
        s=SCALE.get(NAMES[g[0]],1.0); hf=g[5]
        b = "far" if hf < NEAR*s else "near" if hf < CLOSE*s else "close"
        band[g[0]][b][0]+= 1 if j in claimed else 0
        band[g[0]][b][1]+= 1

print(f"\n{'class':<12}" + "".join(f"{x:>18}" for x in ["far (ignored)","near (spoken)","close (urgent)"]))
print("-"*66)
tot=defaultdict(lambda:[0,0])
for c,name in enumerate(NAMES):
    row=f"{name:<12}"
    for b in ["far","near","close"]:
        hit,n = band[c][b]
        tot[b][0]+=hit; tot[b][1]+=n
        row += f"{(f'{hit/n:.3f} (n={n})' if n else '-- '):>18}"
    print(row)
print("-"*66)
print(f"{'ALL':<12}" + "".join(f"{(f'{tot[b][0]/tot[b][1]:.3f} (n={tot[b][1]})' if tot[b][1] else '--'):>18}" for b in ["far","near","close"]))
