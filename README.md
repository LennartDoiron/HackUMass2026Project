# Trashbot

AI recycling bin (HackUMass). A Pi camera looks down at a tray, decides
recyclable / trash / "ask for help", and a servo tilts the tray to the right side.
Voice commands can tilt it too. Everything runs offline on a Raspberry Pi 5.

## Run

    cd ~/trashbot && source venv/bin/activate
    python live_boxes_v2.py          # then open http://trashbot.local:8000

Say **"Trash bot, empty trash"** or **"Trash bot, empty recycling"** to tilt by voice.

## Files

| File | What it is | Test it alone |
|---|---|---|
| `live_boxes_v2.py` | Main program: camera, item finder, AI, voting. Settings at the top. | `python test_v2.py` (no camera needed) |
| `servo.py` | Tray tilt. Settings: flat angle, tilt angle, which side is trash. | `python servo.py` |
| `voice.py` | Voice commands. Settings: wake word, phrases, mic. | `python voice.py` |
| `capture_photos.py` | Take example photos of your items (big accuracy boost). | |
| `check_photos.py` | Score the AI on those photos. | |
| `export_onnx.py` | Makes `models/clip_vision.onnx` (2x faster AI). | |
| `servo_test.py`, `check_audio.py` | Original hardware tests for the servo and mic. | |
| `old/` | Earlier experiments: v1 `live_boxes.py`, TrashNet/YOLO scripts. | |

## Setup on a fresh Pi (models are not stored in git)

    python -m venv --system-site-packages venv && source venv/bin/activate
    pip install torch transformers opencv-python onnx onnxruntime vosk rpi-hardware-pwm
    python -c "from transformers import CLIPModel, CLIPTokenizer as T; n='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(n); T.from_pretrained(n)"   # needs internet once (set HF_HUB_OFFLINE=0)
    python export_onnx.py
    cd models && curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && unzip vosk-model-small-en-us-0.15.zip

`/boot/firmware/config.txt` needs:

    dtparam=i2s=on
    dtoverlay=googlevoicehat-soundcard
    dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4
