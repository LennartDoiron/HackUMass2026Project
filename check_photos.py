"""
check_photos.py - how well does the AI do on your own photos? No camera needed.

    python check_photos.py

For every photo in my_photos/ it hides that one photo, classifies it using the
text descriptions + all the OTHER photos, and checks the bin. Run it after
capture_photos.py. It tells you which items are weak (take more/varied photos
of those, or improve their descriptions in ITEMS).
"""
import numpy as np
import live_boxes_v2 as v2

clf = v2.Classifier()
if clf.photo_feats is None:
    raise SystemExit("No photos yet. Run capture_photos.py first.")

WANT = {"recyclable": "RECYCLE", "trash": "TRASH", "other": "HELP", "nothing": "NOTHING"}
feats, items = clf.photo_feats, clf.photo_item

print(f"\n{'item':16s} {'photos':>6s} {'text only':>10s} {'with photos':>12s}   mistakes (with photos)")
total = [0, 0, 0]
for i in sorted(set(items.tolist())):
    idx = np.where(items == i)[0]
    want = WANT[v2.BIN_OF[i]]
    text_ok, both_ok, wrong = 0, 0, []
    for j in idx:
        saved, clf.photo_feats = clf.photo_feats, None           # text only
        text_ok += v2.decide(clf.item_probs(feats[j]))[0] == want
        clf.photo_feats = saved
        kind, name, conf = v2.decide(clf.item_probs(feats[j], skip_photo=j))
        both_ok += kind == want
        if kind != want:
            wrong.append(f"{kind}:{name}")
    total = [total[0] + len(idx), total[1] + text_ok, total[2] + both_ok]
    print(f"{v2.NAMES[i]:16s} {len(idx):6d} {text_ok / len(idx):10.0%} {both_ok / len(idx):12.0%}   "
          + ", ".join(sorted(set(wrong))))
print(f"{'ALL':16s} {total[0]:6d} {total[1] / total[0]:10.0%} {total[2] / total[0]:12.0%}")

# How alike are photos of the same item vs different items? (for PHOTO_SIM_* settings)
sims = feats @ feats.T
np.fill_diagonal(sims, -1)
same = np.array([sims[j][items == items[j]].max() for j in range(len(items))])
print(f"\nClosest same-item photo similarity: typical {np.median(same):.2f}, "
      f"lowest 10% {np.percentile(same, 10):.2f}   "
      f"(PHOTO_SIM_HIGH={v2.PHOTO_SIM_HIGH} should be at or below the 'lowest 10%' number)")
if len(set(items.tolist())) > 1:
    other = np.array([sims[j][items != items[j]].max() for j in range(len(items))])
    print(f"Closest other-item photo similarity: typical {np.median(other):.2f}")
