"""
test_v2.py - self-test for live_boxes_v2.py. Needs NO camera.

    python test_v2.py

Checks: the model loads offline, how long one classification takes, and that the
item-finder behaves on fake scenes (shadow, lighting drift, touching items...).
The fake scenes are drawings, so PASS here does not replace testing with the
real camera - it just catches broken settings quickly.
"""
import time
import cv2, numpy as np
import live_boxes_v2 as v2

W, H = v2.FRAME_SIZE
rng = np.random.default_rng(0)
ok_all = True


def check(name, got, want):
    global ok_all
    ok = got == want
    ok_all &= ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, expected {want}")


def noisy(img):
    """Add camera-like noise."""
    return np.clip(img.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)


def empty_tray():
    """Light surface, a bit darker towards the corners (like a small lamp)."""
    yy, xx = np.mgrid[0:H, 0:W]
    shade = 1 - 0.25 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    return (np.dstack([200 * shade, 205 * shade, 210 * shade])).astype(np.uint8)


def soft_shadow(img, cx, cy, r, strength=0.72):
    """Darken a blurry circle (no colour change) - like a hand's shadow."""
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (cx, cy), r, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 12)
    return (img * (1 - (1 - strength) * m[..., None])).astype(np.uint8)


# ---------------------------------------------------------------- detector
print("DETECTOR (fake scenes)")
tray = empty_tray()
det = v2.Detector()
det.set_background([noisy(tray) for _ in range(10)])
count = lambda img: len(det.find(noisy(img))[0])

check("empty tray", count(tray), 0)
check("soft shadow only", count(soft_shadow(tray, 300, 240, 90)), 0)
check("lights 8% brighter", count(np.clip(tray * 1.08, 0, 255).astype(np.uint8)), 0)
check("lights 10% dimmer", count((tray * 0.90).astype(np.uint8)), 0)

s = tray.copy(); cv2.rectangle(s, (200, 150), (330, 300), (40, 90, 200), -1)
check("one coloured item", count(s), 1)
check("item + shadow beside it", count(soft_shadow(s, 380, 230, 70)), 1)

s2 = s.copy(); cv2.circle(s2, (480, 330), 55, (60, 160, 60), -1)
check("two separate items", count(s2), 2)

s = tray.copy(); cv2.circle(s, (250, 240), 60, (50, 50, 190), -1); cv2.circle(s, (365, 240), 60, (50, 150, 50), -1)
check("two touching cans", count(s), 2)

s = tray.copy(); cv2.rectangle(s, (150, 200), (400, 290), (190, 120, 40), -1); cv2.rectangle(s, (400, 225), (450, 265), (190, 120, 40), -1)
check("one bottle (body + neck) stays one", count(s), 1)

s = tray.copy(); cv2.rectangle(s, (180, 190), (430, 280), (150, 160, 165), 3); cv2.line(s, (200, 235), (410, 235), (240, 245, 250), 2)
check("clear bottle (outline + glare only)", count(s), 1)

s = tray.copy(); cv2.ellipse(s, (320, 240), (120, 70), 20, 0, 360, (245, 250, 252), -1)
check("white napkin on light surface", count(s), 1)

s = tray.copy(); cv2.rectangle(s, (260, 180), (330, 320), (25, 25, 25), -1)
check("black charger", count(s), 1)

t0 = time.time()
for _ in range(20):
    det.find(tray)
print(f"  detector speed: {(time.time() - t0) / 20 * 1000:.0f} ms per frame")

# ---------------------------------------------------------------- decision
print("DECISION RULES")
def probs(**kw):
    p = np.zeros(len(v2.NAMES))
    for k, val in kw.items():
        p[v2.NAMES.index(k)] = val
    return p
check("clear can", v2.decide(probs(can=0.9, chip_bag=0.1))[0], "RECYCLE")
check("clear chip bag", v2.decide(probs(chip_bag=0.85, can=0.15))[0], "TRASH")
check("50/50 -> help", v2.decide(probs(paper=0.5, napkin=0.5))[0], "HELP")
check("charger -> help", v2.decide(probs(charger=0.8, can=0.2))[0], "HELP")
check("empty -> nothing", v2.decide(probs(empty=0.7, paper=0.3))[0], "NOTHING")

# ---------------------------------------------------------------- classifier
print("CLASSIFIER")
clf = v2.Classifier()
crop = noisy(np.full((200, 200, 3), 128, np.uint8))
clf.classify(crop)
t0 = time.time()
for i in range(8):
    p = clf.classify(crop, turns=i)
ms = (time.time() - t0) / 8 * 1000
print(f"  one classification: {ms:.0f} ms ({clf.backend})")
print(f"  => final answer about {v2.SETTLE_SECONDS + v2.MIN_VOTES * ms / 1000:.1f} s (sure) to "
      f"{v2.SETTLE_SECONDS + v2.MAX_VOTES * ms / 1000:.1f} s (unsure) after the item stops moving")
check("probabilities add up to 1", round(float(p.sum()), 3), 1.0)
check("square crop at frame edge is square", v2.square_crop(tray, (600, 10, 40, 200)).shape[:2], (248, 248))

print("\nALL PASSED" if ok_all else "\nSOME TESTS FAILED - see FAIL lines above")
