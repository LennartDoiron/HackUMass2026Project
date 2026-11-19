from rpi_hardware_pwm import HardwarePWM
from time import sleep, monotonic
import math

# ---------- settings ----------
LEVEL = -20        # your flat position
TILT = 40          # degrees to tilt each way
TILT_TIME = 1.0   # seconds to tilt down (bigger = slower, smoother)
RETURN_TIME = 1.5  # seconds to come back up
SLIDE_WAIT = 1.0   # pause so the item slides off
PWM_CHIP = 0       # if you saw pwmchip2 in step 1, change this to 2

pwm = HardwarePWM(pwm_channel=0, hz=50, chip=PWM_CHIP)   # GPIO12

def duty(angle):
    """Convert -90..90 degrees to a duty cycle (0.5 ms .. 2.5 ms pulse)."""
    angle = max(-90, min(90, angle))
    pulse_ms = 0.5 + (angle + 90) / 180 * 2.0
    return pulse_ms / 20 * 100

current = LEVEL
pwm.start(duty(LEVEL))
sleep(0.5)

def move_smooth(target, duration):
    """Glide to target using a smooth S-curve, updated every 20 ms."""
    global current
    start, t0 = current, monotonic()
    while True:
        t = min(1.0, (monotonic() - t0) / duration)
        eased = t * t * t * (t * (t * 6 - 15) + 10)   # extra-smooth curve
        pwm.change_duty_cycle(duty(start + (target - start) * eased))
        if t >= 1.0:
            break
        sleep(0.02)
    current = target

def sort(direction):
    """direction = +1 for one bin, -1 for the other."""
    move_smooth(LEVEL + direction * TILT, TILT_TIME)
    sleep(SLIDE_WAIT)
    move_smooth(LEVEL, RETURN_TIME)

# ---------- test ----------
print("sorting to side A")
sort(-1)
sleep(1)
print("sorting to side B")
sort(+1)
print("done - holding flat")
sleep(2)
pwm.stop()