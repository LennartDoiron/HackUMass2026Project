"""
voice.py - offline voice commands from the I2S mic ("Trashbot, empty trash").

How it works: `arecord` streams the mic, Vosk (a small offline speech model in
models/vosk-model-small-en-us-0.15) turns it into words. Vosk is told to listen
ONLY for the phrases below, which makes it fast and hard to confuse.

Test on its own:   python voice.py        (prints what it hears; Ctrl+C to stop)

Used by live_boxes_v2.py like this:
    voice.listen(on_command)      # on_command("trash") is called when you say
                                  # "trash bot empty trash", etc.
"""
import json, subprocess, threading
from pathlib import Path
import numpy as np

# ---------- settings ----------
WAKE_WORD = "trash bot"          # must be said first. (Two words: "trashbot" is not
                                 # in the speech model's dictionary.)
# what you say after the wake word  ->  the action sent to the main program
COMMANDS = {
    "empty trash":     "trash",
    "trash":           "trash",
    "garbage":         "trash",
    "empty recycling": "recycle",
    "recycling":       "recycle",
    "recycle":         "recycle",
    "reset":           "reset",     # re-learn the empty background
    # teaching: the item on the tray (or the one just sorted) is saved as an
    # example photo for that bin, so the AI gets it right next time
    "that's recycling": "learn_recycle",
    "this is recycling": "learn_recycle",
    "that's trash":     "learn_trash",
    "this is trash":    "learn_trash",
    "wrong":            "wrong",     # the item just sorted went in the wrong bin
}
MIC_DEVICE  = "plughw:CARD=sndrpigooglevoi"   # the I2S mic (see: arecord -l)
MIC_CHANNEL = 0                  # the mic is on the left channel; right is silent
MODEL_DIR   = Path(__file__).resolve().parent / "models" / "vosk-model-small-en-us-0.15"


def find_command(text):
    """'trash bot empty trash' -> 'trash'.  Anything else -> None."""
    text = text.replace("[unk]", " ")
    text = " ".join(text.split())
    if not text.startswith(WAKE_WORD):
        return None
    return COMMANDS.get(text[len(WAKE_WORD):].strip())


def make_recognizer():
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)                                  # quiet
    phrases = [f"{WAKE_WORD} {c}" for c in COMMANDS] + ["[unk]"]   # [unk] = anything else
    return KaldiRecognizer(Model(str(MODEL_DIR)), 16000, json.dumps(phrases))


def to_16k_mono(raw):
    """48 kHz stereo 16-bit bytes from arecord -> 16 kHz mono bytes for Vosk."""
    a = np.frombuffer(raw, np.int16).reshape(-1, 2)[:, MIC_CHANNEL]
    return a[:len(a) // 3 * 3].reshape(-1, 3).mean(axis=1).astype(np.int16).tobytes()


def listen(on_command, on_heard=None):
    """Start listening in the background. Never crashes the main program: if the
    mic or model is missing it prints why and voice is simply off."""
    def run():
        try:
            rec = make_recognizer()
            mic = subprocess.Popen(
                ["arecord", "-q", "-D", MIC_DEVICE, "-c", "2", "-r", "48000",
                 "-f", "S16_LE", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Voice NOT available ({e}).")
            return
        print(f'Voice ready: say "{WAKE_WORD}, empty trash" / "{WAKE_WORD}, empty recycling".')
        while True:
            raw = mic.stdout.read(19200)             # 0.1 s of audio
            if len(raw) < 19200:
                print("Voice stopped: the microphone stream ended (is it plugged in / in use?).")
                return
            if rec.AcceptWaveform(to_16k_mono(raw)): # True = a phrase just finished
                text = json.loads(rec.Result())["text"]
                command = find_command(text)
                if on_heard and text.replace("[unk]", "").strip():
                    on_heard(text, command)
                if command:
                    on_command(command)

    threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    import time
    listen(lambda c: print("   ==> COMMAND:", c),
           lambda text, c: print(f'heard: "{text}"'))
    while True:
        time.sleep(1)
