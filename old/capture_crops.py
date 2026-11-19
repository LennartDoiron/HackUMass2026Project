import sys, time, random, cv2, numpy as np
from pathlib import Path
from picamera2 import Picamera2

label, count = sys.argv[1], int(sys.argv[2])
cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}))
cam.start(); time.sleep(2)
md = cam.capture_metadata()
cam.set_controls({"AeEnable": False, "AwbEnable": False,
                  "ExposureTime": md["ExposureTime"],
                  "AnalogueGain": md["AnalogueGain"],
                  "ColourGains": md["ColourGains"]})
time.sleep(1)

prep = lambda img: cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (21, 21), 0)
input("Clear the area, then press Enter...")
bg = prep(cam.capture_array())
kernel = np.ones((15, 15), np.uint8)
print(f"Capturing {count} '{label}' crops. Move/swap the item between shots.")

saved = 0
while saved < count:
    time.sleep(1)
    frame = cam.capture_array()
    _, mask = cv2.threshold(cv2.absdiff(bg, prep(frame)), 30, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), kernel)
    cs = [c for c in cv2.findContours(mask, cv2.RETR_EXTERNAL,
          cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c) > 1500]
    if len(cs) != 1:
        print("need exactly 1 item in view, skipping"); continue
    x, y, w, h = cv2.boundingRect(cs[0])
    crop = frame[max(0, y-10):y+h+10, max(0, x-10):x+w+10]
    split = "val" if random.random() < 0.2 else "train"
    out = Path("dataset") / split / label
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"me_{label}_{int(time.time()*1000)}.jpg"), crop)
    saved += 1
    print(f"{saved}/{count}")