"""
live_boxes_v2.py - the trashbot main program: camera + AI + tray servo + voice.

Run:   cd ~/trashbot && source venv/bin/activate && python live_boxes_v2.py
View:  http://trashbot.local:8000         live annotated view
       http://trashbot.local:8000/mask    debug: what the item-finder sees (green = item)
       http://trashbot.local:8000/reset   re-learn the empty background (tray must be empty!)

Say:   "Trash bot, empty trash" / "Trash bot, empty recycling" to tilt the tray by voice.

How it works (same order as the code below):
  1. DETECTOR   finds items by comparing each frame to the empty background.
                Shadows are ignored (darker but same colour = shadow).
  2. TRACKER    follows each box between frames and waits until it stops moving.
  3. CLASSIFIER CLIP turns the crop into numbers and compares them with
                  a) text descriptions of each item (works with no photos), and
                  b) YOUR photos in my_photos/<item>/ (optional, much more accurate).
  4. VOTING     several frames are averaged before a final answer is shown.
  5. ACTION     RECYCLE / TRASH answer -> the tray tilts that way (servo.py).
                A voice command (voice.py) tilts it too, e.g. after "ASK FOR HELP".

Other files:
  servo.py           tray tilt (settings: angles, which side is trash)   python servo.py = test
  voice.py           voice commands (settings: phrases, mic)             python voice.py = test
  capture_photos.py  take your own example photos        check_photos.py  score them
  export_onnx.py     makes the fast model file (2x speed) test_v2.py       self-test, no camera
  old/               earlier experiments (v1 live_boxes.py, TrashNet/YOLO scripts, old tests)
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # never touch the internet (demo is offline)
import time, threading, warnings
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).resolve().parent

# =============================================================================
#  SETTINGS - everything you might want to change is in this section
# =============================================================================

# ---- 1. ITEMS: what the bin knows, and which bin each item goes in ----------
# Format:   "item_name": ("bin", ["description", "another description", ...]),
#   bin = "recyclable" | "trash" | "other" (not for either bin -> ask for help)
#         | "nothing" (false alarm, e.g. a shadow -> ignored)
# To move an item to another bin, change the first word. To add an item, add a
# line. The item_name is also the photo folder name: my_photos/<item_name>/.
# Rules below follow Amherst, MA curbside recycling.
ITEMS = {
    # ---------------- recyclable ----------------
    "plastic_bottle": ("recyclable", ["a plastic bottle", "an empty plastic water bottle",
                                      "a plastic soda bottle", "a clear plastic bottle"]),
    "can":            ("recyclable", ["an aluminum can", "a soda can", "a metal drink can",
                                      "a tin food can"]),
    "glass":          ("recyclable", ["a glass bottle", "a glass jar"]),
    "paper":          ("recyclable", ["a sheet of paper", "a piece of printer paper",
                                      "a crumpled sheet of paper", "a newspaper or flyer",
                                      "an envelope"]),
    "cardboard":      ("recyclable", ["a piece of cardboard", "a flattened cardboard box",
                                      "a small cardboard box", "brown corrugated cardboard"]),
    # Empty paper cups are accepted in Massachusetts recycling since 2026
    # (RecycleSmartMA). Change to "trash" if your judges/hauler say otherwise.
    "paper_cup":      ("recyclable", ["a paper coffee cup", "a disposable paper cup",
                                      "a takeaway coffee cup"]),
    "plastic_cup":    ("recyclable", ["a clear plastic cup", "an empty plastic drink cup"]),
    # ---------------- trash ----------------
    "chip_bag":       ("trash", ["a chip bag", "a bag of potato chips", "a shiny foil snack bag",
                                 "an empty crisp packet"]),
    "wrapper":        ("trash", ["a candy wrapper", "a granola bar wrapper",
                                 "a crumpled plastic food wrapper"]),
    "napkin":         ("trash", ["a used paper napkin", "a crumpled tissue",
                                 "a crumpled paper towel"]),
    "plastic_bag":    ("trash", ["a plastic bag", "a plastic grocery bag", "plastic film"]),
    "styrofoam":      ("trash", ["a styrofoam cup", "a styrofoam food container",
                                 "a piece of white foam packaging"]),
    "food":           ("trash", ["food scraps", "leftover food", "a banana peel",
                                 "a piece of fruit"]),
    "greasy":         ("trash", ["a greasy paper plate", "a food-stained paper container",
                                 "a greasy pizza box"]),
    # ---------------- other (ask for help) ----------------
    "charger":        ("other", ["a phone charger", "a USB cable", "a power adapter",
                                 "a tangled cable"]),
    "phone":          ("other", ["a smartphone", "a mobile phone"]),
    "battery":        ("other", ["a battery", "AA batteries"]),
    "electronics":    ("other", ["an electronic device", "earbuds or headphones"]),
    "keys":           ("other", ["a bunch of keys", "a wallet"]),
    "hand":           ("other", ["a human hand", "a person's hand and arm"]),
    # ---------------- nothing (false alarm) ----------------
    "empty":          ("nothing", ["an empty table", "a plain blank surface",
                                   "a shadow on a table"]),
}

# Each description is tried in all of these sentences and the results averaged
# ("prompt ensembling"). They describe YOUR camera view: top-down, on a surface.
TEMPLATES = ["a photo of {}.",
             "a top-down photo of {} lying on a table.",
             "a close-up photo of {}, seen from above.",
             "a photo of {} on a plain white surface."]

# ---- 2. DECISION: when is the bin sure enough? -------------------------------
CONF_NEEDED   = 0.60   # winning bin needs at least this much (0..1), else ask for help
MARGIN_NEEDED = 0.20   # ...and must beat the other bin by this much
MIN_VOTES     = 3      # frames to look at before a final answer is allowed
MAX_VOTES     = 6      # stop voting here and answer (or ask for help)
EARLY_CONF    = 0.85   # after MIN_VOTES, answer immediately if this sure

# ---- 3. YOUR PHOTOS (few-shot). Ignored until my_photos/ has pictures --------
PHOTO_DIR       = HERE / "my_photos"
PHOTO_WEIGHT    = 0.90  # how much to trust your photos over the text descriptions (0..1)
PHOTO_SIM_LOW   = 0.65  # crop looks NOT AT ALL like any photo below this -> text only
PHOTO_SIM_HIGH  = 0.80  # crop looks just like a photo above this -> full PHOTO_WEIGHT
PHOTO_SHARPNESS = 100   # higher = closest photos win more decisively
PHOTO_TOP_K     = 3     # average the K most similar photos of each item

# ---- 4. DETECTOR: finding items against the empty background -----------------
MIN_AREA       = 1500   # ignore blobs smaller than this (pixels)
COLOR_DIFF     = 12     # colour change that counts as "item" (lower = more sensitive)
DARK_RATIO     = 0.55   # darker than this x background = item. Between this and
                        #   ~1.0 with no colour change = SHADOW, ignored.
BRIGHT_RATIO   = 1.12   # brighter than this x background = item (glare, white things)
EDGE_DIFF      = 45     # new sharp edges = item (finds clear plastic). Shadows are soft.
CLOSE_PX       = 13     # glue fragments of one item closer than this (pixels)
SPLIT_TOUCHING = True   # try to split two touching items into two boxes
SPLIT_NECK     = 0.60   # split only if the join is narrower than this x item width
BIG_CHANGE     = 0.60   # more than this fraction of the view changed = lighting/camera moved

# ---- 5. TRACKER: stillness -----------------------------------------------------
SETTLE_SECONDS = 0.35   # item must be still this long before the AI looks at it
MOTION_LIMIT   = 2.5    # average pixel change inside the box that counts as "moving"
LOST_SECONDS   = 0.5    # forget a box after it has been gone this long
EMPTY_REFRESH_SECONDS = 1.5  # tray empty this long -> slowly re-learn background (drift)
ABSORB_SECONDS = 3.0    # a box the AI calls "empty" for this long is merged into background

# ---- 6. MODEL / CAMERA / WEB ---------------------------------------------------
CLIP_NAME  = "openai/clip-vit-base-patch32"
ONNX_FILE  = HERE / "models" / "clip_vision.onnx"   # made by export_onnx.py (2x faster)
AI_THREADS = 3          # of 4 cores; leaves one for the live view
FRAME_SIZE = (640, 480)
PORT       = 8000
WAIT_FOR_ENTER = True   # False = grab the background immediately at startup

# ---- 7. SERVO + VOICE (their own settings are at the top of servo.py / voice.py) ----
SERVO_ENABLED = True    # False = never move the tray (vision only)
VOICE_ENABLED = True    # False = no microphone
AFTER_SORT_SECONDS = 1.0  # after the tray is flat again, wait this long before looking

# =============================================================================
#  END OF SETTINGS
# =============================================================================

NAMES = list(ITEMS)                                   # item names, fixed order
BIN_OF = [ITEMS[n][0] for n in NAMES]                 # bin of each item, same order


# -----------------------------------------------------------------------------
#  CLASSIFIER
# -----------------------------------------------------------------------------
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def square_crop(frame, box, pad=0.12):
    """Cut a SQUARE around the box (plus a little margin). Square matters: CLIP
    only looks at a square, so a tight crop of a long bottle would lose both ends."""
    x, y, w, h = box
    side = int(max(w, h) * (1 + 2 * pad))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = cx - side // 2, cy - side // 2
    H, W = frame.shape[:2]
    # part of the square may be outside the frame -> repeat the edge pixels there
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x0 + side - W), max(0, y0 + side - H)
    crop = frame[max(0, y0):y0 + side, max(0, x0):x0 + side]
    if left or top or right or bottom:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return crop


def to_clip_input(crop_bgr):
    """BGR crop -> the 1x3x224x224 float array CLIP expects."""
    shrink = crop_bgr.shape[0] > 224
    img = cv2.resize(crop_bgr, (224, 224),
                     interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((img - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)[None]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Classifier:
    def __init__(self, use_photos=True):
        import torch
        from transformers import CLIPModel, CLIPTokenizer
        warnings.filterwarnings("ignore")
        print("Loading CLIP...")
        model = CLIPModel.from_pretrained(CLIP_NAME).eval()
        tok = CLIPTokenizer.from_pretrained(CLIP_NAME)

        # --- text side: one averaged "fingerprint" per item (done once) ---
        rows = []
        with torch.no_grad():
            for name in NAMES:
                sentences = [t.format(d) for d in ITEMS[name][1] for t in TEMPLATES]
                out = model.text_model(**tok(sentences, padding=True, return_tensors="pt"))
                f = model.text_projection(out.pooler_output)
                f = f / f.norm(dim=-1, keepdim=True)
                f = f.mean(0)
                rows.append((f / f.norm()).numpy())
            self.text = np.stack(rows)                       # (items, 512)
            self.scale = float(model.logit_scale.exp())      # CLIP's own sharpness (~100)

        # --- image side: fast ONNX file if present, else plain torch ---
        self.backend = "torch"
        if ONNX_FILE.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = AI_THREADS
                self.sess = ort.InferenceSession(str(ONNX_FILE), opts,
                                                 providers=["CPUExecutionProvider"])
                self.backend = "onnx"
                del model                                    # free ~600 MB
            except Exception as e:
                print("ONNX model failed to load, using torch instead:", e)
        if self.backend == "torch":
            print("  (slow path: run  python export_onnx.py  once for 2x speed)")
            torch.set_num_threads(AI_THREADS)
            self.torch, self.vision, self.proj = torch, model.vision_model, model.visual_projection
        print(f"CLIP ready ({self.backend}), {len(NAMES)} items.")

        self.photo_feats, self.photo_item = None, None
        if use_photos:
            self.load_photos()

    # ---- image -> 512 numbers ("embedding"), length 1 ----
    def embed(self, crops_bgr):
        x = np.concatenate([to_clip_input(c) for c in crops_bgr])
        if self.backend == "onnx":
            f = self.sess.run(None, {"x": x})[0]
        else:
            with self.torch.inference_mode():
                f = self.proj(self.vision(pixel_values=self.torch.from_numpy(x)).pooler_output).numpy()
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    # ---- your photos ----
    def load_photos(self):
        """Embed every picture in my_photos/<item>/. Results are cached in
        my_photos/embeddings.npz so only new pictures cost time."""
        cache_file = PHOTO_DIR / "embeddings.npz"
        cache = {}
        if cache_file.exists():
            z = np.load(cache_file, allow_pickle=False)
            cache = dict(zip(z["keys"].tolist(), z["feats"]))
        keys, feats, items, new = [], [], [], 0
        for i, name in enumerate(NAMES):
            for p in sorted((PHOTO_DIR / name).glob("*.jpg")):
                key = f"{name}/{p.name}/{p.stat().st_mtime_ns}"
                if key not in cache:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    h, w = img.shape[:2]                     # make square, just in case
                    cache[key] = self.embed([square_crop(img, (0, 0, w, h), pad=0)])[0]
                    new += 1
                keys.append(key); feats.append(cache[key]); items.append(i)
        if PHOTO_DIR.exists():
            unknown = [d.name for d in PHOTO_DIR.iterdir() if d.is_dir() and d.name not in ITEMS]
            if unknown:
                print("WARNING: photo folders that match no item in ITEMS (ignored):", unknown)
        if not feats:
            print("No photos in my_photos/ -> using text descriptions only.")
            return
        self.photo_feats, self.photo_item = np.stack(feats), np.array(items)
        if new:
            np.savez(cache_file, keys=np.array(keys), feats=self.photo_feats)
        counts = {NAMES[i]: int((self.photo_item == i).sum()) for i in sorted(set(items))}
        print(f"Using {len(feats)} of your photos:", counts)

    # ---- embedding -> probability of each item ----
    def item_probs(self, feat, skip_photo=None):
        """Mix of (a) text match and (b) similarity to your photos.
        skip_photo is only used by check_photos.py (leave-one-out testing)."""
        p_text = softmax(self.scale * (self.text @ feat))
        if self.photo_feats is None:
            return p_text
        sims = self.photo_feats @ feat
        if skip_photo is not None:
            sims = sims.copy(); sims[skip_photo] = -1
        # score of each item = average of its K most similar photos
        score = np.full(len(NAMES), -np.inf)
        for i in set(self.photo_item.tolist()):
            s = np.sort(sims[self.photo_item == i])[-PHOTO_TOP_K:]
            score[i] = s.mean()
        p_photo = softmax(PHOTO_SHARPNESS * score)           # items without photos get 0
        # only trust photos when the crop really resembles one of them
        trust = np.clip((sims.max() - PHOTO_SIM_LOW) / (PHOTO_SIM_HIGH - PHOTO_SIM_LOW), 0, 1)
        w = PHOTO_WEIGHT * trust
        return (1 - w) * p_text + w * p_photo

    def classify(self, crop_bgr, turns=0):
        """One look at one square crop. turns = rotate by turns x 90 degrees
        (the camera looks straight down, so any rotation is a valid view)."""
        if turns % 4:
            crop_bgr = np.ascontiguousarray(np.rot90(crop_bgr, turns % 4))
        return self.item_probs(self.embed([crop_bgr])[0])


def decide(probs):
    """Item probabilities -> (kind, item, confidence).
    kind: "RECYCLE", "TRASH", "HELP" (ask a human) or "NOTHING" (false alarm)."""
    bins = {"recyclable": 0.0, "trash": 0.0, "other": 0.0, "nothing": 0.0}
    for p, b in zip(probs, BIN_OF):
        bins[b] += float(p)
    best = max(bins, key=bins.get)
    # name the most likely item INSIDE the winning bin, so label and bin agree
    item = max((i for i in range(len(NAMES)) if BIN_OF[i] == best), key=lambda i: probs[i])
    item, conf = NAMES[item].replace("_", " "), bins[best]
    if best == "nothing":
        return "NOTHING", item, conf
    if best == "other":
        return "HELP", item, conf
    margin = abs(bins["recyclable"] - bins["trash"])
    if conf < CONF_NEEDED or margin < MARGIN_NEEDED:
        return "HELP", item + "?", conf
    return ("RECYCLE" if best == "recyclable" else "TRASH"), item, conf


COLORS = {"RECYCLE": (255, 120, 0), "TRASH": (60, 60, 255), "HELP": (0, 200, 255),
          "NOTHING": (120, 120, 120), None: (200, 200, 200)}


# -----------------------------------------------------------------------------
#  DETECTOR - background subtraction that ignores shadows and lighting drift
# -----------------------------------------------------------------------------
def _ellipse(px):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


class Detector:
    def set_background(self, frames):
        """frames = a few pictures of the EMPTY tray (averaged to remove noise)."""
        mean = np.mean(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)
        self.L, self.A, self.B, edge = self._features(mean)
        # background edges, fattened a little so tiny camera wobble is forgiven
        self.edge = cv2.dilate(edge, _ellipse(5))

    @staticmethod
    def _features(frame):
        """Brightness (L), colour (A, B) and edge strength of a frame."""
        lab = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB)
        L, A, B = [c.astype(np.float32) for c in cv2.split(lab)]
        L += 1.0                                             # avoid divide-by-zero
        edge = cv2.magnitude(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1))
        return L, A, B, edge

    def find(self, frame):
        """Returns (boxes, mask, big_change). boxes = [(x, y, w, h), ...]."""
        L, A, B, edge = self._features(frame)
        ratio = L / self.L                                   # 1.0 = same brightness
        # Lighting drift: if the WHOLE picture got 5% brighter, undo that first.
        gain = float(np.clip(np.median(ratio[::8, ::8]), 0.6, 1.6))
        ratio /= gain
        colour = cv2.magnitude(A - self.A, B - self.B)
        item = ((colour > COLOR_DIFF) |                      # different colour
                (ratio < DARK_RATIO) |                       # too dark to be a shadow
                (ratio > BRIGHT_RATIO) |                     # brighter (glare, white item)
                (edge - self.edge * gain > EDGE_DIFF))       # new sharp edge (clear plastic)
        mask = item.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(3))          # remove specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(CLOSE_PX))  # join fragments
        # fill holes (a clear bottle is often only an outline), then remove thin
        # leftover lines (e.g. the soft edge of a shadow)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask[:] = 0
        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(7))

        big_change = cv2.countNonZero(mask) > BIG_CHANGE * mask.size
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            box = cv2.boundingRect(c)
            boxes += split_touching(mask, box) if SPLIT_TOUCHING else [box]
        return boxes, mask, big_change

    def refresh(self, frame, rate=0.1, box=None):
        """Blend the current frame into the background (whole frame, or one box).
        Used when the tray is empty, so slow lighting changes never build up."""
        L, A, B, edge = self._features(frame)
        edge = cv2.dilate(edge, _ellipse(5))
        s = np.s_[:, :] if box is None else np.s_[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        for bg, new in ((self.L, L), (self.A, A), (self.B, B), (self.edge, edge)):
            bg[s] = (1 - rate) * bg[s] + rate * new[s]


def split_touching(mask, box):
    """If a blob looks like two items joined by a narrow neck, return two boxes.
    Method: measure how 'thick' the blob is everywhere (distance transform). Two
    touching items give two thick centres with a thin join between them."""
    x, y, w, h = box
    blob = np.zeros((h + 2, w + 2), np.uint8)                # 1px empty border
    blob[1:-1, 1:-1] = mask[y:y + h, x:x + w]
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    thickest = dist.max()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (dist > SPLIT_NECK * thickest).astype(np.uint8))
    # keep only cores that are themselves reasonably thick and big (ignore crumbs)
    cores = [i for i in range(1, n)
             if dist[labels == i].max() > 0.6 * thickest and stats[i, cv2.CC_STAT_AREA] > 30]
    if len(cores) < 2:
        return [box]
    # grow each core back out to the blob edge (watershed) and box each part
    markers = np.zeros(blob.shape, np.int32)
    markers[blob == 0] = 1
    for k, i in enumerate(cores):
        markers[labels == i] = k + 2
    relief = cv2.cvtColor(255 - cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                          cv2.COLOR_GRAY2BGR)
    cv2.watershed(relief, markers)
    out = []
    for k in range(len(cores)):
        ys, xs = np.where(markers == k + 2)
        if len(xs) >= MIN_AREA / 2:
            out.append((x + int(xs.min()) - 1, y + int(ys.min()) - 1,
                        int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1))
    return out if len(out) >= 2 else [box]


# -----------------------------------------------------------------------------
#  TRACKER - one Track per item on the tray
# -----------------------------------------------------------------------------
def overlap(a, b):
    """How much two boxes overlap, 0..1 (intersection over union)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / (aw * ah + bw * bh - iw * ih)


class Track:
    def __init__(self, box, now):
        self.box = box
        self.still_since = now      # when it last moved
        self.last_seen = now
        self.reset()

    def reset(self):
        """Item moved or changed -> throw away the votes and start again."""
        self.votes = []             # one probability list per frame looked at
        self.final = None           # (kind, item, conf) once the vote is done
        self.final_time = None
        self.version = getattr(self, "version", 0) + 1   # lets the AI thread spot a reset

    def label(self):
        if self.final:
            kind, item, conf = self.final
            text = {"RECYCLE": "RECYCLE", "TRASH": "TRASH", "HELP": "ASK FOR HELP",
                    "NOTHING": "nothing"}[kind]
            return f"{text}: {item} {conf:.0%}", COLORS[kind]
        if self.votes:
            return f"checking {len(self.votes)}/{MAX_VOTES}...", COLORS[None]
        return "hold still...", COLORS[None]


def update_tracks(tracks, boxes, motion_of, now):
    """Match this frame's boxes to existing tracks; make/forget tracks as needed."""
    unused = list(tracks)
    for box in boxes:
        best = max(unused, key=lambda t: overlap(t.box, box), default=None)
        if best is None or overlap(best.box, box) < 0.3:
            tracks.append(Track(box, now))                   # new item
            continue
        unused.remove(best)
        moved = overlap(best.box, box) < 0.7 or motion_of(box) > MOTION_LIMIT
        best.box, best.last_seen = box, now
        if moved:
            best.still_since = now
            if best.votes or best.final:
                best.reset()
    tracks[:] = [t for t in tracks if now - t.last_seen < LOST_SECONDS]


# -----------------------------------------------------------------------------
#  MAIN PROGRAM - camera, the two loops, and the web stream
# -----------------------------------------------------------------------------
def open_camera():
    """Start the Pi camera and lock exposure + white balance (same as v1)."""
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"}))      # "RGB888" arrives as BGR
    cam.start()
    time.sleep(2)                                            # let auto-exposure settle
    md = cam.capture_metadata()
    cam.set_controls({"AeEnable": False, "AwbEnable": False,
                      "ExposureTime": md["ExposureTime"],
                      "AnalogueGain": md["AnalogueGain"],
                      "ColourGains": md["ColourGains"]})
    time.sleep(1)
    return cam


def main():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    clf = Classifier()
    clf.classify(np.zeros((224, 224, 3), np.uint8))          # warm-up run
    cam = open_camera()
    det = Detector()
    if WAIT_FOR_ENTER:
        input("Clear the area under the camera, then press Enter...")
    det.set_background([cam.capture_array() for _ in range(10)])
    print(f"Background saved. Open http://trashbot.local:{PORT} on your Mac.")

    lock = threading.Lock()
    state = {"frame": None, "jpg": None, "mask_jpg": None, "reset": False}
    tracks = []

    # ---------- actions: tilt the tray (from the AI or from a voice command) ----------
    tray = None
    if SERVO_ENABLED:
        import servo
        tray = servo.Tray()

    def do_action(action, why):
        """action = "trash" / "recycle" (tilt that way) or "reset" (re-learn background)."""
        if action == "reset":
            state["reset"] = True
            print(f"{why}: re-learning the background")
        elif tray is None:
            print(f"{why}: would tilt to {action} (SERVO_ENABLED is False)")
        elif tray.sort(action):
            print(f"{why}: tilting to {action}")
        else:
            print(f"{why}: ignored, the tray is already moving")

    def on_final_answer(kind, item, confidence):
        """Called ONCE per item when the vote is finished and exactly one item is
        in view. kind is "RECYCLE", "TRASH" or "HELP" (HELP = don't move; wait
        for a person or a voice command)."""
        print(f"FINAL: {kind:8s} {item}  ({confidence:.0%})")
        if kind in ("RECYCLE", "TRASH"):
            do_action("recycle" if kind == "RECYCLE" else "trash", "AI")

    if VOICE_ENABLED:
        import voice
        voice.listen(lambda command: do_action(command, "VOICE"),
                     lambda text, command: print(f'heard: "{text}"'))

    # ---------- fast loop: camera + detector + tracker + drawing (no AI) ----------
    def video_loop():
        history = []                 # small grey frames from the last few frames
        empty_since, n, fps, t_fps = time.time(), 0, 0.0, time.time()
        look_again_at, sorting = 0.0, ""
        while True:
            frame = cam.capture_array()
            now = time.time()
            n += 1

            # While the tray is tilting (and just after) the whole picture changes,
            # so don't look for items: forget the old boxes and just show the camera.
            if tray and tray.busy:
                look_again_at, sorting = now + AFTER_SORT_SECONDS, tray.side
            if now < look_again_at:
                with lock:
                    tracks.clear()
                history, empty_since = [], now
                cv2.putText(frame, f"Sorting -> {sorting.upper()}", (10, FRAME_SIZE[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                state["jpg"] = cv2.imencode(".jpg", frame)[1].tobytes()
                continue
            if state["reset"]:                               # /reset was opened
                det.set_background([frame] + [cam.capture_array() for _ in range(9)])
                with lock:
                    tracks.clear()
                state["reset"] = False
                continue

            boxes, mask, big_change = det.find(frame)

            # movement = difference to the frame from ~0.15 s ago, at quarter size
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None,
                               fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            history = (history + [small])[-4:]
            change = cv2.absdiff(small, history[0])

            def motion_of(box):
                x, y, w, h = [v // 4 for v in box]
                return float(change[y:y + h + 1, x:x + w + 1].mean())

            with lock:
                state["frame"] = frame
                update_tracks(tracks, boxes, motion_of, now)
                # A box the AI has called "nothing" for a while is a false alarm
                # (lighting changed there) -> merge it into the background.
                for t in tracks:
                    if (t.final and t.final[0] == "NOTHING"
                            and now - t.final_time > ABSORB_SECONDS):
                        det.refresh(frame, rate=1.0, box=t.box)
                view = [(t.box, *t.label()) for t in tracks]
                n_items = sum(not (t.final and t.final[0] == "NOTHING") for t in tracks)

            # tray empty for a while -> slowly follow lighting drift
            if boxes:
                empty_since = now
            elif now - empty_since > EMPTY_REFRESH_SECONDS and n % 5 == 0:
                det.refresh(frame)

            # ----- draw -----
            out = frame.copy()
            for (x, y, w, h), text, color in view:
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
                cv2.putText(out, text, (x, max(25, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            status = ("Big change - is the tray empty? open /reset" if big_change else
                      "Waiting for item" if n_items == 0 else
                      "1 item" if n_items == 1 else
                      f"{n_items} items - please insert one at a time")
            cv2.putText(out, status, (10, FRAME_SIZE[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            if n % 20 == 0:
                fps, t_fps = 20 / (now - t_fps), now
            cv2.putText(out, f"{fps:.0f} fps", (FRAME_SIZE[0] - 90, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            state["jpg"] = cv2.imencode(".jpg", out)[1].tobytes()
            if n % 3 == 0:
                # debug view: camera picture dimmed, "item" pixels tinted green
                dbg = frame // 3
                dbg[mask > 0] = frame[mask > 0] // 2 + np.array([0, 120, 0], np.uint8)
                cv2.putText(dbg, "DEBUG /mask: green = detected as item", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                state["mask_jpg"] = cv2.imencode(".jpg", dbg)[1].tobytes()

    # ---------- slow loop: AI looks at still items and collects votes ----------
    def ai_loop():
        while True:
            now = time.time()
            with lock:
                ready = [t for t in tracks if not t.final
                         and now - t.still_since >= SETTLE_SECONDS
                         and now - t.last_seen < 0.2]
                if not ready:
                    job = None
                else:
                    t = min(ready, key=lambda t: len(t.votes))   # fewest votes first
                    job = (t, t.version, len(t.votes), square_crop(state["frame"], t.box).copy())
            if job is None:
                time.sleep(0.03)
                continue
            t, version, turns, crop = job
            probs = clf.classify(crop, turns)                # ~0.12 s, outside the lock
            with lock:
                if t.version != version:                     # it moved meanwhile: discard
                    continue
                t.votes.append(probs)
                kind, item, conf = decide(np.mean(t.votes, axis=0))
                enough = len(t.votes) >= MIN_VOTES and conf >= EARLY_CONF and kind != "HELP"
                if enough or len(t.votes) >= MAX_VOTES:
                    t.final, t.final_time = (kind, item, conf), time.time()
                    # "alone" = the only real item (false alarms don't count)
                    alone = sum(not (x.final and x.final[0] == "NOTHING") for x in tracks) == 1
                else:
                    continue
            if alone and kind != "NOTHING":
                on_final_answer(kind, item, conf)

    # ---------- web stream ----------
    class Stream(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/reset"):
                state["reset"] = True
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Background re-learned. Go back to the live view.\n")
                return
            key = "mask_jpg" if self.path.startswith("/mask") else "jpg"
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = state[key]
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
    ThreadingHTTPServer(("0.0.0.0", PORT), Stream).serve_forever()


if __name__ == "__main__":
    main()
