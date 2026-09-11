# Tests for merge_datasets.py using tiny fake datasets built in a temp folder.
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_datasets as m


def test_remap_label_line_translates_class_id():
    source_names = ["Stairs", "Person", "Trash"]
    mapping = {"Stairs": "stairs_up", "Person": "person"}
    line = "0 0.5 0.5 0.2 0.3"
    out = m.remap_label_line(line, source_names, mapping)
    assert out == f"{m.TARGET_CLASSES.index('stairs_up')} 0.5 0.5 0.2 0.3"


def test_remap_label_line_drops_unmapped_class():
    assert m.remap_label_line("2 0.5 0.5 0.2 0.3", ["a", "b", "Trash"], {"a": "person"}) is None


def test_split_is_deterministic_and_roughly_80_10_10():
    names = [m.split_name(i, 1000, seed=42) for i in range(1000)]
    again = [m.split_name(i, 1000, seed=42) for i in range(1000)]
    assert names == again
    assert 750 <= names.count("train") <= 850
    assert 50 <= names.count("val") <= 150
    assert 50 <= names.count("test") <= 150


def _fake_dataset(root: Path, name: str, class_names, files):
    """files: list of (filename, image_bytes, label_text)"""
    d = root / name
    (d / "train" / "images").mkdir(parents=True)
    (d / "train" / "labels").mkdir(parents=True)
    (d / "data.yaml").write_text("names: " + json.dumps(class_names) + "\nnc: %d\n" % len(class_names))
    for fname, img_bytes, label in files:
        (d / "train" / "images" / fname).write_bytes(img_bytes)
        (d / "train" / "labels" / (Path(fname).stem + ".txt")).write_text(label)


def test_merge_remaps_dedupes_and_writes_yaml(tmp_path: Path):
    raw = tmp_path / "raw"; raw.mkdir()
    maps = tmp_path / "mappings"; maps.mkdir()
    out = tmp_path / "merged"
    _fake_dataset(raw, "dsA", ["Stairs", "Person"], [
        ("a1.jpg", b"IMG1", "0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.1 0.1\n"),
        ("a2.jpg", b"IMG2", "1 0.5 0.5 0.2 0.3\n"),
    ])
    _fake_dataset(raw, "dsB", ["car", "Stairs"], [
        ("b1.jpg", b"IMG1", "0 0.5 0.5 0.2 0.3\n"),   # same bytes as a1 -> duplicate, dropped
        ("b2.jpg", b"IMG3", "1 0.5 0.5 0.2 0.3\n"),
        ("b3.jpg", b"IMG4", "5 0.5 0.5 0.2 0.3\n"),   # bad class id -> line dropped, image dropped (no labels left)
    ])
    (maps / "dsA.json").write_text(json.dumps({"Stairs": "stairs_up", "Person": "person"}))
    (maps / "dsB.json").write_text(json.dumps({"car": "car", "Stairs": "stairs_up"}))

    counts = m.merge(raw, maps, out, seed=1)

    kept = sorted(p.name for s in ("train", "val", "test") for p in (out / "images" / s).glob("*.jpg"))
    assert kept == ["dsA_a1.jpg", "dsA_a2.jpg", "dsB_b2.jpg"]
    yaml_text = (out / "data.yaml").read_text()
    assert "stairs_up" in yaml_text and "train: images/train" in yaml_text
    total_stairs = sum(counts[s].get("stairs_up", 0) for s in counts)
    assert total_stairs == 2
