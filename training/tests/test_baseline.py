"""
Purpose : Tests for make_baseline_labels.py — translating OUR class ids into COCO class ids
          so the un-tuned yolo11n.pt can be scored on our test split.
Depends : pytest. Imports the script under test from training/scripts.
Question you should be able to answer: why is a class COCO does not know dropped instead of kept?
Usage   : pytest training/tests/test_baseline.py -q
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_baseline_labels as b

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog"]  # first 17 is enough for the test
OURS = ["person", "dog", "car", "bicycle", "motorbike", "stairs_up"]


def test_overlapping_class_is_translated():
    assert b.remap_to_coco_line("4 0.5 0.5 0.1 0.1", OURS, COCO) == "3 0.5 0.5 0.1 0.1"   # motorbike -> motorcycle


def test_new_class_is_dropped():
    assert b.remap_to_coco_line("5 0.5 0.5 0.1 0.1", OURS, COCO) is None   # stairs_up: COCO can't know it
