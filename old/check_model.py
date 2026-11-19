from ultralytics import YOLO
from pathlib import Path
m = YOLO("runs/classify/train/weights/best.pt")
for c in ["recyclable", "trash"]:
    files = list(Path("dataset/val", c).glob("*.jpg"))
    right = 0
    for f in files:
        r = m(str(f), verbose=False)[0]
        right += r.names[r.probs.top1] == c
    print(f"{c}: {right}/{len(files)} correct")