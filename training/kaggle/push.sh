#!/usr/bin/env bash
# Purpose : Push train_kaggle.py to Kaggle as a private T4 script kernel for one job.
# Usage   : training/kaggle/push.sh v2 | v3 | ablations
#           then: uvx kaggle kernels status divyabasavraj25/vision-<job>
#                 uvx kaggle kernels output divyabasavraj25/vision-<job> -p training/runs/kaggle_<job>
set -euo pipefail
JOB="${1:?usage: push.sh v2|v3|ablations}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
SCRIPTS="$HERE/../scripts"
# Set JOB and embed the current runner scripts (see SCRIPTS in train_kaggle.py).
python3 - "$HERE/train_kaggle.py" "$TMP/train_kaggle.py" "$JOB" "$SCRIPTS" <<'PY'
import sys, pathlib
src, dst, job, scripts = sys.argv[1:]
files = {f: (pathlib.Path(scripts) / f).read_text() for f in ("run_ablations.py", "oversample.py")}
out = []
for line in pathlib.Path(src).read_text().splitlines(keepends=True):
    if line.startswith("JOB = "):
        line = f"JOB = {job!r}\n"
    elif line.startswith("SCRIPTS: dict[str, str] = {}"):
        line = f"SCRIPTS: dict[str, str] = {files!r}\n"
    out.append(line)
pathlib.Path(dst).write_text("".join(out))
PY
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
