"""
Purpose : Download Roboflow Universe datasets in YOLO format into training/data/raw/<short-name>/
          so merge_datasets.py can find <short-name>/data.yaml. Needs a free Roboflow account key.
Depends : roboflow (pip), python-dotenv optional — key is read from $ROBOFLOW_API_KEY or training/.env.
Question you should be able to answer: why do we download in YOLO format and not COCO JSON?
Usage   : python training/scripts/download_roboflow.py <short-name>=<workspace>/<project>/<version> [...]
          e.g. python training/scripts/download_roboflow.py ood=fpn/ood-pbnro/1 blind=obstacles-for-blind-zjnnn/obstacles-for-blind/2
"""
from __future__ import annotations
import os, sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    env = Path(__file__).resolve().parents[1] / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY (env or training/.env). Get it at https://app.roboflow.com/settings/api")
    return key


def download(spec: str, key: str) -> Path:
    name, ref = spec.split("=", 1)
    workspace, project, version = ref.strip("/").split("/")
    from roboflow import Roboflow
    dest = RAW / name
    if (dest / "data.yaml").exists():
        print(f"{name}: already present, skipping")
        return dest
    print(f"{name}: downloading {workspace}/{project} v{version} ...")
    Roboflow(api_key=key).workspace(workspace).project(project).version(int(version)).download("yolov11", location=str(dest))
    print(f"{name}: done -> {dest}")
    return dest


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    api_key = load_key()
    RAW.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[1:]:
        download(spec, api_key)
