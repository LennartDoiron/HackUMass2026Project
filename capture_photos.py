"""
capture_photos.py - take example photos of YOUR items with YOUR camera.

    python capture_photos.py <item_name> [how_many]
    python capture_photos.py plastic_bottle 15

<item_name> must be one of the names in ITEMS at the top of live_boxes_v2.py.
Photos are saved to my_photos/<item_name>/ and live_boxes_v2.py picks them up
automatically the next time it starts.

How to use it:
  1. Clear the tray, press Enter (background is saved).
  2. Put ONE item under the camera and take your hand away.
     When it has been still for half a second a photo is taken ("saved 1/15").
  3. Move / rotate / flip / swap the item and let go again -> next photo.
     It will not take a second photo until the item has moved.
Uses exactly the same item-finder and crop as the live program, so the photos
look like what the AI sees during the demo.
"""
import sys, time
import cv2, numpy as np
import live_boxes_v2 as v2

if len(sys.argv) < 2 or sys.argv[1] not in v2.ITEMS:
    print(__doc__)
    print("Item names:", ", ".join(v2.NAMES))
    sys.exit(1)
item = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 15
folder = v2.PHOTO_DIR / item
folder.mkdir(parents=True, exist_ok=True)

cam = v2.open_camera()
det = v2.Detector()
input("Clear the area under the camera, then press Enter...")
det.set_background([cam.capture_array() for _ in range(10)])
print(f"Taking {count} photos of '{item}'. Put one item down and take your hand away.")

saved, still_since, last_small, need_move = 0, None, None, False
while saved < count:
    frame = cam.capture_array()
    boxes, _, _ = det.find(frame)
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA)
    moving = last_small is not None and cv2.absdiff(small, last_small).mean() > 1.0
    last_small = small
    now = time.time()
    if moving or len(boxes) != 1:
        still_since, need_move = None, False      # something moved: ready for a new photo
        continue
    if need_move:                                 # same pose as the last photo: wait
        continue
    still_since = still_since or now
    if now - still_since < 0.5:
        continue
    crop = v2.square_crop(frame, boxes[0])
    cv2.imwrite(str(folder / f"{item}_{int(now * 1000)}.jpg"), crop)
    saved += 1
    need_move = True
    print(f"saved {saved}/{count}  -> now move, rotate or swap the item")

print(f"Done. {len(list(folder.glob('*.jpg')))} photos in {folder}")
print("Look through them and delete any bad ones (hand in shot, item cut off).")
