"""Step-by-step diagnostic for an LX-16A that twitches but won't move.

Separates the three usual culprits:
  1. position sensor (potentiometer) - can the servo read its own angle?
  2. motor / gearbox             - can it spin freely with position control bypassed?
  3. position control            - does a small move actually get there?

REMOVE THE HORN / LEG LINKAGE FIRST so the servo is unloaded - test 2 spins it continuously.

Run with:
    python3 servo_diag.py                         # lists ports / prompts you to pick one
    python3 servo_diag.py /dev/cu.usbserial-110    # or pass the port directly
"""
import sys
import time

from pylx16a.lx16a import LX16A

from test1 import SERVO_ID, pick_port


def sample(servo, seconds, label):
    """Print angle and supply voltage every 0.2 s for `seconds`."""
    end = time.time() + seconds
    while time.time() < end:
        print(f"  {label}: angle {servo.get_physical_angle():7.1f}   vin {servo.get_vin() / 1000.0:.2f}V")
        time.sleep(0.2)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else pick_port()
    LX16A.initialize(port)
    servo = LX16A(SERVO_ID, disable_torque=True)

    print("\n=== Config stored on the servo ===")
    print(f"  physical angle : {servo.get_physical_angle():.1f}  (should be within 0-240)")
    print(f"  angle limits   : {servo.get_angle_limits()}")
    print(f"  angle offset   : {servo.get_angle_offset()}")
    print(f"  vin limits (mV): {servo.get_vin_limits()}   current vin: {servo.get_vin()} mV")
    print(f"  temp limit / now: {servo.get_temp_limit()} / {servo.get_temp()} C")
    print(f"  motor mode     : {servo.is_motor_mode()}")

    try:
        input("\n=== Test 1: position sensor ===\nTorque is OFF. Press Enter, then slowly turn the "
              "output shaft by hand for 10 s...")
        sample(servo, 10, "by hand")
        print("  -> Angle should change smoothly as you turn. Stuck, jumping, or reading outside "
              "0-240 = bad/misaligned position sensor.")

        input("\n=== Test 2: motor + gears (motor mode, no position feedback) ===\n"
              "Make sure nothing is attached to the shaft. Press Enter to spin each way for 2 s...")
        servo.enable_torque()
        servo.motor_mode(400)
        sample(servo, 2, "spin +")
        servo.motor_mode(-400)
        sample(servo, 2, "spin -")
        servo.servo_mode()
        print("  -> If it spun both ways, the motor and gears are fine. If it just buzzed, "
              "the motor or gearbox is jammed/damaged.")

        input("\n=== Test 3: small position move ===\nPress Enter to move 20 degrees toward the middle...")
        start = servo.get_physical_angle()
        target = max(10.0, min(230.0, start + (20 if start < 120 else -20)))
        print(f"  moving {start:.1f} -> {target:.1f}")
        servo.move(target, time=500)
        sample(servo, 1.5, "move")
        print(f"  -> Final angle {servo.get_physical_angle():.1f} vs target {target:.1f}.")
    except KeyboardInterrupt:
        pass
    finally:
        servo.servo_mode()
        servo.disable_torque()
        print("\nTorque disabled, exiting.")


if __name__ == "__main__":
    main()
