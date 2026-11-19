from picamera2 import Picamera2
import time

cam = Picamera2()
cam.configure(cam.create_still_configuration(main={"size": (1280, 960)}))
cam.start()
time.sleep(2)  # let exposure settle
cam.capture_file("from_python.jpg")
cam.stop()
print("Saved from_python.jpg")
