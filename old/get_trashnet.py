import random
from pathlib import Path
from datasets import load_dataset

ds = load_dataset("garythung/trashnet", split="train")
names = ds.features["label"].names
recyclable = {"glass", "paper", "cardboard", "plastic", "metal"}

per_class_cap = 80          # keeps recyclable from swamping trash
counts = {}
for i, ex in enumerate(ds):
    name = names[ex["label"]]
    if name in recyclable and counts.get(name, 0) >= per_class_cap:
        continue
    counts[name] = counts.get(name, 0) + 1
    target = "recyclable" if name in recyclable else "trash"
    split = "val" if random.random() < 0.2 else "train"
    folder = Path("dataset") / split / target
    folder.mkdir(parents=True, exist_ok=True)
    ex["image"].convert("RGB").resize((320, 240)).save(folder / f"tn_{i}.jpg")

print("Done:", counts)