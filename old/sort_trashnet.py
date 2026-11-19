import random
from pathlib import Path
from PIL import Image

src = next(p.parent for p in Path("trashnet_raw").rglob("cardboard")
           if "__MACOSX" not in p.parts)
recyclable = {"glass", "paper", "cardboard", "plastic", "metal"}
cap = 80   # keeps recyclable from swamping trash
counts = {}

for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
    name = class_dir.name
    if name not in recyclable and name != "trash":
        continue
    files = sorted(class_dir.glob("*.jpg"))
    random.shuffle(files)
    if name in recyclable:
        files = files[:cap]
    target = "recyclable" if name in recyclable else "trash"
    for f in files:
        split = "val" if random.random() < 0.2 else "train"
        out = Path("dataset") / split / target
        out.mkdir(parents=True, exist_ok=True)
        Image.open(f).convert("RGB").resize((320, 240)).save(
            out / f"tn_{name}_{f.stem}.jpg")
    counts[name] = len(files)

print("Done:", counts)