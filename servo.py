"""
servo.py - tilts the tray: one way for trash, the other way for recycling.

Same hardware and motion as servo_test.py (hardware PWM on GPIO12, smooth S-curve).

Test on its own:   python servo.py            (tilts trash side, then recycling side)
                   python servo.py trash      (just one side)
Find true flat:    python servo.py level -20  (holds that angle; try -18, -22... until the
                   tray is perfectly flat, then put that number in LEVEL below)

Used by live_boxes_v2.py like this:
    tray = Tray()
    tray.sort("trash")      # or "recycle". Returns at once; the tilt runs in the background.
    tray.busy               # True while the tray is moving
"""
import atexit, threading, time

# ---------- settings ----------
LEVEL        = 10     # flat position (degrees) - found with: python servo.py level 10
TILT         = 70     # degrees to tilt each way
TRASH_SIDE   = -1     # -1 or +1. If trash goes into the recycling bin, swap these two.
RECYCLE_SIDE = +1
TILT_TIME    = 1.0    # seconds to tilt down (bigger = slower, smoother)
SLIDE_WAIT   = 1.0    # pause so the item slides off
RETURN_TIME  = 1.5    # seconds to come back up
PWM_CHIP     = 0      # /sys/class/pwm/pwmchip0
PWM_CHANNEL  = 0      # channel 0 = GPIO12


def duty(angle):
    """Convert -90..90 degrees to a duty cycle (0.5 ms .. 2.5 ms pulse at 50 Hz)."""
    angle = max(-90, min(90, angle))
    pulse_ms = 0.5 + (angle + 90) / 180 * 2.0
    return pulse_ms / 20 * 100


class Tray:
    def __init__(self):
        self.busy = False               # True while tilting (vision pauses meanwhile)
        self.side = None                # "trash" / "recycle" while busy
        self._lock = threading.Lock()
        self._angle = LEVEL
        try:
            from rpi_hardware_pwm import HardwarePWM
            self.pwm = HardwarePWM(pwm_channel=PWM_CHANNEL, hz=50, chip=PWM_CHIP)
            self.pwm.start(duty(LEVEL))             # go to flat and hold there
            atexit.register(self.pwm.stop)
            print("Servo ready (holding flat).")
        except Exception as e:
            self.pwm = None                         # no servo: just print what would happen
            print(f"Servo NOT available ({e}) - tilts will only be printed.")

    def sort(self, side):
        """Start a tilt towards "trash" or "recycle". Returns False (and does
        nothing) if the tray is already moving."""
        with self._lock:
            if self.busy:
                return False
            self.busy, self.side = True, side
        threading.Thread(target=self._run, args=(side,), daemon=True).start()
        return True

    def _run(self, side):
        try:
            direction = TRASH_SIDE if side == "trash" else RECYCLE_SIDE
            self._move(LEVEL + direction * TILT, TILT_TIME)
            time.sleep(SLIDE_WAIT)
            self._move(LEVEL, RETURN_TIME)
        finally:
            self.busy, self.side = False, None

    def _move(self, target, duration):
        """Glide to target using a smooth S-curve, updated every 20 ms."""
        start, t0 = self._angle, time.monotonic()
        while True:
            t = min(1.0, (time.monotonic() - t0) / duration)
            eased = t * t * t * (t * (t * 6 - 15) + 10)
            if self.pwm:
                self.pwm.change_duty_cycle(duty(start + (target - start) * eased))
            if t >= 1.0:
                break
            time.sleep(0.02)
        self._angle = target


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["level"]:                  # calibration: hold one angle
        LEVEL = float(sys.argv[2]) if len(sys.argv) > 2 else LEVEL
        Tray()
        print(f"Holding {LEVEL} degrees. Is the tray flat? Ctrl+C to stop.")
        while True:
            time.sleep(1)
    tray = Tray()
    time.sleep(0.5)
    for side in sys.argv[1:] or ["trash", "recycle"]:
        print("tilting to the", side, "side")
        tray.sort(side)
        while tray.busy:
            time.sleep(0.1)
        time.sleep(1)
    print("done")
