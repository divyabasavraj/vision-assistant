#!/usr/bin/env bash
# Purpose : Push train_kaggle.py to Kaggle as a private T4 script kernel for one job.
# Usage   : training/kaggle/push.sh v2 | ablations
#           then: uvx kaggle kernels status divyabasavraj25/vision-<job>
#                 uvx kaggle kernels output divyabasavraj25/vision-<job> -p training/runs/kaggle_<job>
set -euo pipefail
JOB="${1:?usage: push.sh v2|ablations}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
sed "s/^JOB = .*/JOB = \"$JOB\"/" "$HERE/train_kaggle.py" > "$TMP/train_kaggle.py"
cat > "$TMP/kernel-metadata.json" <<EOF
{
  "id": "divyabasavraj25/vision-$JOB",
  "title": "vision-$JOB",
  "code_file": "train_kaggle.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["divyabasavraj25/vision-assistant-merged"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF
uvx kaggle kernels push -p "$TMP" --accelerator NvidiaTeslaT4
