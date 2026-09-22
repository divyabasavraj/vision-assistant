#!/usr/bin/env bash
# Purpose : Run the whole of Task 9's training overnight, one job at a time, on the Mac's GPU (MPS).
#           Each job writes training/runs/<name>_test.json; a job that fails does not stop the queue.
# Usage   : caffeinate -imsu training/scripts/run_queue.sh   (keep the Mac plugged in, lid open)
set -u
cd "$(dirname "$0")/../.."
PY=training/.venv/bin/python
LOGS=training/runs/logs
mkdir -p "$LOGS"

run () {  # run <name> <args...>
  local name="$1"; shift
  if [ -f "training/runs/${name}_test.json" ]; then
    echo "[$(date +%H:%M)] $name already done, skipping"; return 0
  fi
  echo "[$(date +%H:%M)] START $name"
  "$PY" training/scripts/run_ablations.py --name "$name" "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M)] END   $name (exit $rc)"
  return $rc
}

# 1. The final candidate: continue from v1's weights, which were still improving at epoch 50.
run n640_v2 --base training/results/n640_v1/best.pt --epochs 30 --batch 32

# 2-4. Ablation arms. Identical budget (12 epochs), identical start (COCO yolo11n.pt), batch 16 so
#      that the yolo11s arm fits in 16 GB of unified memory and every arm shares the same batch.
run n640_e12        --epochs 12 --batch 16                      # control
run n320_e12        --epochs 12 --batch 16 --imgsz 320          # ablation A: image size
run n640_noaug_e12  --epochs 12 --batch 16 --no-aug             # ablation C: augmentation

# 5. Ablation B last: yolo11s is ~3x the compute of nano, and the most likely to run out of memory.
run s640_e12 --base yolo11s.pt --epochs 12 --batch 16 || {
  echo "[$(date +%H:%M)] s640_e12 failed at batch 16, retrying at batch 8"
  run s640_e12_b8 --base yolo11s.pt --epochs 12 --batch 8
}

echo "[$(date +%H:%M)] QUEUE FINISHED"
ls -1 training/runs/*_test.json
