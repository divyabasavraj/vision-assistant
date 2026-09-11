One JSON file per folder in training/data/raw/, same name as the folder.
Keys = class names exactly as written in that dataset's data.yaml.
Values = one of our TARGET_CLASSES (see scripts/merge_datasets.py):
  person, dog, car, bicycle, motorbike, auto, bus,
  stairs_up, stairs_down, curb, pole, chair, table, door, crosswalk
Leave out any class you don't want; its labels are dropped.

Example — training/mappings/stairs-v3.json:
{ "Stair Up": "stairs_up", "Stair Down": "stairs_down", "Person": "person" }
