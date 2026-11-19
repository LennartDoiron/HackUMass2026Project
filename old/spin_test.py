from gpiozero import Servo
from time import sleep

servo = Servo(12, min_pulse_width=0.0005, max_pulse_width=0.0025)

SPEED = 0.3        # 0 to 1; slower = gentler and more consistent
TILT_TIME = 0.8    # long enough to reach the stop every time
RETURN_TIME = 0.4  # tune this until the tray ends up level

def sort(direction):        # +1 = one bin, -1 = the other
    servo.value = SPEED * direction
    sleep(TILT_TIME)
    servo.detach()          # stop
    sleep(1.0)              # let the item slide off
    servo.value = -SPEED * direction
    sleep(RETURN_TIME)
    servo.detach()          # stop at level

sort(+1)
sleep(2)
sort(-1)