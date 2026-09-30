"""
Purpose : Tests for oversample.py — repeat-factor sampling of the train split.
Depends : pytest. Imports the script under test from training/scripts.
Question you should be able to answer: why is a class seen in every image never repeated?
          Its frequency f_c is above the threshold, so sqrt(t / f_c) < 1 and the factor floors at 1.
Usage   : pytest training/tests/test_oversample.py -q
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import oversample as o


def test_rare_class_image_gets_the_larger_factor():
    # class 0 in all 10 images, class 1 in one: f_1 = 0.1, t = 0.4 -> r = sqrt(4) = 2
    classes = {f"i{k}": {0} for k in range(9)} | {"rare": {0, 1}}
    r = o.repeat_factors(classes, threshold=0.4)
    assert r["rare"] == 2.0
    assert r["i0"] == 1.0


def test_factor_is_capped():
    classes = {f"i{k}": {0} for k in range(999)} | {"rare": {1}}
    assert o.repeat_factors(classes, threshold=0.5, cap=4.0)["rare"] == 4.0


def test_write_list_repeats_rare_image(tmp_path: Path):
    for split in ("images", "labels"):
        (tmp_path / split / "train").mkdir(parents=True)
    for k in range(9):
        (tmp_path / "images" / "train" / f"c{k}.jpg").write_bytes(b"")
        (tmp_path / "labels" / "train" / f"c{k}.txt").write_text("0 .5 .5 .1 .1\n")
    (tmp_path / "images" / "train" / "rare.jpg").write_bytes(b"")
    (tmp_path / "labels" / "train" / "rare.txt").write_text("1 .5 .5 .1 .1\n")

    dst = tmp_path / "list.txt"
    inst = o.write_list(tmp_path, threshold=0.4, dst=dst)

    lines = dst.read_text().split()
    assert sum(l.endswith("rare.jpg") for l in lines) == 2
    assert len(lines) == 11
    assert inst[1] == 2 and inst[0] == 9
