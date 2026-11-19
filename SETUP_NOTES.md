# TrashBot – setup notes (backup of the HackUMass build)

An AI recycling bin. A camera looks down at a tray and decides **recycle / trash /
ask for help**, and a servo tilts the tray into the right bin. You can also give it
voice commands. Everything runs offline on a Raspberry Pi 5.

The setup recorded here: Raspberry Pi OS (Debian 13 "trixie", 64-bit),
kernel 6.18, Python 3.13.5.

---

## 1. Hardware

| Part | What we used |
|---|---|
| Computer | Raspberry Pi 5 (8 GB) |
| Camera | Pi Camera v1.3 (OV5647, fixed focus), pointing straight down 20–25 cm above the tray |
| Tray motor | Hobby servo, driven by hardware PWM |
| Microphone | I2S MEMS microphone (mono, uses the left channel) |
| Speaker | Bluetooth speaker (paired by hand – not used by the code yet) |
| Light | Small lamp over the tray (the purple tray is dark, so more light = better detection) |
| Power | Official 27 W USB-C supply (5 V / 5 A) recommended – see the warning below |

## 2. Wiring

Pin numbers are the **physical** pin numbers on the 40-pin header (pin 1 = 3.3 V,
the corner nearest the SD card side).

### Servo (tray tilt)
| Servo wire | Connect to |
|---|---|
| Signal (yellow/orange) | **GPIO12** – physical pin **32** (hardware PWM0 channel 0) |
| Power (red) | **Separate 5–6 V supply** (strongly recommended) |
| Ground (brown/black) | Ground of that supply **and** a Pi GND pin (e.g. pin **34**) – the grounds must be shared |

⚠️ Powering the servo from the Pi's 5 V pin made the Pi switch itself off (red LED)
when the servo moved while the AI was busy. Use a separate supply.

GPIO13 (pin 33) is also set up as a second PWM channel by the overlay but isn't used.

### I2S microphone
| Mic pin | Connect to |
|---|---|
| VDD / 3V | 3.3 V – pin **1** |
| GND | GND – pin **6** |
| SCK / BCLK (bit clock) | **GPIO18** – pin **12** |
| WS / LRCL (word select) | **GPIO19** – pin **35** |
| SD / DOUT (data) | **GPIO20** – pin **38** |
| L/R (if present) | GND → left channel (the code reads channel 0) |

The Pi uses the `googlevoicehat-soundcard` driver for it. It shows up as
card `sndrpigooglevoi` in `arecord -l`. It only records properly as **16-bit**
(`-f S16_LE`), and only the left channel has sound.

### Camera
Plug the ribbon cable into the **CAM/DISP 0** connector. The Pi 5 has the small
22-pin connector, so the v1.3 camera needs a 22-pin → 15-pin adapter cable.

## 3. config.txt lines we added

The full original is saved as `config_backup.txt`. These are the lines that matter
(in `/boot/firmware/config.txt`):

```
dtparam=i2s=on                                      # enable I2S (for the mic)
dtoverlay=googlevoicehat-soundcard                  # I2S mic driver (under [all])
dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4   # hardware PWM on GPIO12/13 (under [all])
```

Everything else in that file is the Raspberry Pi OS default.

## 4. Rebuild on a fresh Pi, step by step

1. **Flash the SD card** with Raspberry Pi Imager: *Raspberry Pi OS (64-bit)*.
   In the Imager settings, set the hostname to `trashbot`, the user to `admin`,
   turn on SSH, and add your Wi-Fi.
2. **Boot and update:**
   ```
   sudo apt update && sudo apt full-upgrade -y
   sudo apt install -y python3-picamera2 python3-venv alsa-utils
   ```
3. **Copy the backup over from your Mac and unpack it:**
   ```
   scp ~/Desktop/trashbot_backup.tar.gz admin@trashbot.local:~
   ssh admin@trashbot.local
   tar -xzf trashbot_backup.tar.gz        # creates ~/trashbot
   ```
4. **Hardware settings:** add the three lines from section 3 to
   `/boot/firmware/config.txt` (`sudo nano /boot/firmware/config.txt`), then
   `sudo reboot`. Check them:
   - `arecord -l` → should list `sndrpigooglevoi`
   - `ls /sys/class/pwm` → should show `pwmchip0`
   - `rpicam-hello --list-cameras` → should list `ov5647`
5. **Python environment.** Use `--system-site-packages` so the venv can see Picamera2:
   ```
   cd ~/trashbot
   python3 -m venv --system-site-packages venv
   source venv/bin/activate
   pip install -r requirements-venv-only.txt
   ```
   Use `requirements-venv-only.txt`, **not** `requirements.txt`. The full
   `requirements.txt` also lists packages that came from `apt` (picamera2,
   python-apt, PyGObject…), which `pip` can't install. It's kept as a record only.
   Never use `sudo pip`.
6. **AI models.** `models/` is in the backup:
   - `models/clip_vision.onnx` (fast image half of CLIP)
   - `models/vosk-model-small-en-us-0.15` (offline speech)

   The text half of CLIP still comes from the Hugging Face cache. If the cache
   wasn't backed up, download it once (needs internet):
   ```
   HF_HUB_OFFLINE=0 python -c "from transformers import CLIPModel, CLIPTokenizer as T; n='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(n); T.from_pretrained(n)"
   ```
   If `models/clip_vision.onnx` is ever missing, `python export_onnx.py` recreates it.
7. **Bluetooth speaker** (the code doesn't use it yet, but to pair it again):
   ```
   bluetoothctl
     power on
     agent on
     scan on            # wait for the speaker's address (XX:XX:XX:XX:XX:XX)
     pair XX:XX:XX:XX:XX:XX
     trust XX:XX:XX:XX:XX:XX
     connect XX:XX:XX:XX:XX:XX
     scan off
     quit
   ```
   `trust` makes it reconnect by itself after a reboot.
8. **Test each part:**
   ```
   python test_v2.py              # AI + detector self-test, no camera needed
   python servo.py level 10       # tray should be flat; Ctrl+C to stop
   python servo.py                # tilts trash side, then recycling side
   python voice.py                # say "Trash bot, empty trash"; Ctrl+C to stop
   ```
9. **Start at every power-on:** `./install_service.sh` (asks for your password).

## 5. The main program

**`live_boxes_v2.py` is the main program.** It runs the camera, item detection,
the AI, the tray servo and voice, and serves a live view at
**http://trashbot.local:8000** (`/mask` = debug view, `/reset` = re-learn the empty tray).

- **Automatic (normal use):** after `install_service.sh`, it starts at power-on.
  Keep the tray empty while it starts – it tilts each way once to calibrate.
  - Live log: `journalctl -u trashbot -f`
  - Start/crash history: `cat logs/crash.log`
- **By hand:**
  ```
  sudo systemctl stop trashbot          # the service holds the camera
  cd ~/trashbot && source venv/bin/activate
  python live_boxes_v2.py
  ```

Voice commands always start with "Trash bot", then:
- "empty trash" / "empty recycling" – tilt the tray
- "that's recycling" / "that's trash" – teach it the item on the tray
- "wrong" – the item it just sorted went in the wrong bin
- "reset" – re-learn the empty tray

### Settings we tuned (in case they're lost)
- `servo.py`: `LEVEL = 10` (true flat for our servo horn), `TILT = 70`,
  `TRASH_SIDE = -1`, `RECYCLE_SIDE = +1`
- All the AI and detector settings are in the SETTINGS section at the top of
  `live_boxes_v2.py`. The item → bin rules (Amherst, MA) are in `ITEMS`.
- `my_photos/<item>/` holds the example photos the AI learns from. Bad lessons
  were moved to `my_photos_rejected/`.

## 6. Other files

| File / folder | What it is |
|---|---|
| `servo.py`, `voice.py` | Tray servo and voice modules (settings at the top of each) |
| `capture_photos.py`, `check_photos.py` | Take example photos / score the AI on them |
| `export_onnx.py`, `test_v2.py` | Rebuild the fast model / self-test |
| `trashbot.service`, `install_service.sh` | Auto-start at power-on |
| `servo_test.py`, `check_audio.py` | Original hardware tests |
| `old/` | Earlier experiments (v1 `live_boxes.py`, TrashNet/YOLO scripts) |
| `dataset/`, `runs/`, `trashnet_raw/`, `dataset-resized.zip`, `yolo11n-cls.pt` | The first (failed) YOLO/TrashNet attempt – not needed to run TrashBot |
| `requirements.txt` | Full `pip freeze` (record only) |
| `requirements-venv-only.txt` | What to `pip install` on a rebuild |
| `config_backup.txt` | Copy of the original `/boot/firmware/config.txt` |
