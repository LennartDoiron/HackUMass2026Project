import time, threading, cv2, numpy as np, torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from picamera2 import Picamera2
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# ---------- settings ----------
CONF_NEEDED = 0.60     # below this -> "ask for help"
MIN_AREA = 1500        # ignore specks smaller than this (pixels)
torch.set_num_threads(4)

PROMPTS = {
    "a plastic bottle": "recyclable",
    "an aluminum soda can": "recyclable",
    "a glass bottle or jar": "recyclable",
    "paper or cardboard": "recyclable",
    "a chip bag or snack wrapper": "trash",
    "a used napkin or tissue": "trash",
    "food waste": "trash",
    "a styrofoam cup or container": "trash",
    "a phone charger or cable": "other",
    "a battery": "other",
    "a phone or electronic device": "other",
    "keys": "other",
    "a human hand": "other",
}
NAMES = list(PROMPTS)

# ---------- AI model: load once, pre-compute the text part once ----------
print("Loading CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def as_tensor(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out

with torch.no_grad():
    t = proc(text=[f"a photo of {n}" for n in NAMES],
             return_tensors="pt", padding=True)
    text_feats = as_tensor(model.get_text_features(**t))
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    scale = model.logit_scale.exp()

def classify(crop_bgr):
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        f = as_tensor(model.get_image_features(**proc(images=img, return_tensors="pt")))
        f = f / f.norm(dim=-1, keepdim=True)
        probs = (scale * f @ text_feats.T).softmax(dim=-1)[0].tolist()
    scores = {"recyclable": 0.0, "trash": 0.0, "other": 0.0}
    for name, p in zip(NAMES, probs):
        scores[PROMPTS[name]] += p
    best = max(scores, key=scores.get)
    return best, scores[best], NAMES[int(np.argmax(probs))]

def describe(label, conf, what):
    if conf < CONF_NEEDED or label == "other":
        return f"ASK FOR HELP: {what}", (0, 200, 255)
    if label == "recyclable":
        return f"RECYCLE: {what}", (255, 120, 0)
    return f"TRASH: {what}", (60, 60, 255)

# ---------- camera ----------
cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}))
cam.start()
time.sleep(2)
md = cam.capture_metadata()
cam.set_controls({"AeEnable": False, "AwbEnable": False,
                  "ExposureTime": md["ExposureTime"],
                  "AnalogueGain": md["AnalogueGain"],
                  "ColourGains": md["ColourGains"]})
time.sleep(1)

def prep(img):
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (21, 21), 0)

input("Clear the area under the camera, then press Enter...")
background = prep(cam.capture_array())
print("Background saved. Open http://trashbot.local:8000 on your Mac.")

lock = threading.Lock()
latest_frame, latest_boxes, results, latest_jpg = None, [], [], None
kernel = np.ones((15, 15), np.uint8)

def same_box(a, b):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = a, b
    close = abs((x1 + w1/2) - (x2 + w2/2)) < 40 and abs((y1 + h1/2) - (y2 + h2/2)) < 40
    similar = abs(w1 - w2) < 0.3 * max(w1, w2) and abs(h1 - h2) < 0.3 * max(h1, h2)
    return close and similar

def find_result(box, res):
    for r in res:
        if same_box(box, r[0]):
            return r
    return None

# ---------- fast loop: camera + boxes + drawing (no AI here) ----------
def video_loop():
    global latest_frame, latest_boxes, latest_jpg, results
    while True:
        frame = cam.capture_array()
        _, mask = cv2.threshold(cv2.absdiff(background, prep(frame)), 30, 255,
                                cv2.THRESH_BINARY)
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours
                 if cv2.contourArea(c) > MIN_AREA]
        with lock:
            latest_frame, latest_boxes = frame, boxes
            if not boxes:
                results = []          # tray empty -> forget old answers
            res = list(results)

        out = frame.copy()
        for b in boxes:
            r = find_result(b, res)
            text, color = (r[1], r[2]) if r else ("checking...", (200, 200, 200))
            x, y, w, h = b
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
            cv2.putText(out, text, (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        status = ("Waiting for item" if not boxes else
                  "1 item" if len(boxes) == 1 else
                  f"{len(boxes)} items - please insert one at a time")
        cv2.putText(out, status, (10, 470), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        ok, jpg = cv2.imencode(".jpg", out)
        with lock:
            latest_jpg = jpg.tobytes()

# ---------- slow loop: AI only runs for NEW boxes ----------
def ai_loop():
    global results
    while True:
        with lock:
            frame, boxes, res = latest_frame, list(latest_boxes), list(results)
        if frame is None or all(find_result(b, res) for b in boxes):
            time.sleep(0.1)
            continue
        new = []
        for b in boxes:
            r = find_result(b, res)
            if r is None:
                x, y, w, h = b
                crop = frame[max(0, y-10):y+h+10, max(0, x-10):x+w+10]
                text, color = describe(*classify(crop))
                r = (b, text, color)
            new.append(r)
        with lock:
            results = new

# ---------- web stream ----------
class Stream(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with lock:
                    jpg = latest_jpg
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                     + jpg + b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass
    def log_message(self, *args):
        pass

threading.Thread(target=video_loop, daemon=True).start()
threading.Thread(target=ai_loop, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", 8000), Stream).serve_forever()