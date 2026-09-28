import time

import legoeducation as le

colorsensor = le.ColorSensor()
print("Connecting to Color Sensor...")
colorsensor.connect()
print("Connected. Move a hand/robot in front of the sensor. Ctrl+C to stop.\n")

try:
    while True:
        sensor = colorsensor.sensor
        print(f"reflection={sensor.reflection:6.1f}  value={sensor.value:6.1f}  "
              f"rawRGB=({sensor.rawRed:5.0f}, {sensor.rawGreen:5.0f}, {sensor.rawBlue:5.0f})")
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    colorsensor.disconnect()
