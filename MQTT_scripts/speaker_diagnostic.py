import time

import legoeducation as le

motor = le.DoubleMotor()
print("Connecting to Double Motor...")
motor.connect()
print("Connected. Sweeping through frequencies, listen for each one.\n")

for frequency in range(220, 2701, 200):
    print(f"Playing {frequency} Hz...")
    motor.beep(frequency=frequency, count=1, blocking=True)
    time.sleep(0.3)

motor.disconnect()
print("\nDone. Note which frequencies (if any) you could actually hear.")
