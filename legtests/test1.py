"""Basic LX-16A servo control over a BusLinker USB adapter.

Lists available serial ports (so you don't have to guess which one the
BusLinker enumerated as), connects to it, reads the servo's input voltage
and temperature as a quick sanity check, then lets you drive the servo by
typing an angle at a prompt.

Run with:
    python3 test1.py                          # lists ports / prompts you to pick one
    python3 test1.py /dev/cu.usbserial-110     # or pass the port directly
"""
import sys
import time

import serial
import serial.tools.list_ports
from pylx16a.lx16a import LX16A, ServoChecksumError, ServoTimeoutError

SERVO_ID = 8  # the LX-16A's bus ID - 1 is the factory default for a single new/unconfigured servo
MOVE_TIME_MS = 1000  # how long each commanded move should take - smoother than an instant jump

# What a lost connection looks like: an unplugged servo cable or dead motor power stops the
# servo replying (timeout, or a garbled reply -> checksum error); an unplugged BusLinker makes
# the serial port itself fail.
DISCONNECT_ERRORS = (ServoTimeoutError, ServoChecksumError, serial.SerialException, OSError)


class ServoDisconnected(Exception):
    pass


def pick_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found - is the BusLinker plugged into this computer via micro-USB?")
        sys.exit(1)
    if len(ports) == 1:
        print(f"Using the only serial port found: {ports[0].device}")
        return ports[0].device
    print("Multiple serial ports found:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  ({p.description})")
    choice = input("Which one is the BusLinker? Enter a number: ").strip()
    return ports[int(choice)].device


def ask_angle(prompt):
    """Prompt until the user enters a number in the servo's 0-240 range."""
    while True:
        raw = input(prompt).strip()
        try:
            angle = float(raw)
        except ValueError:
            print("Not a number - try again.")
            continue
        if 0 <= angle <= 240:
            return angle
        print("Angle must be between 0 and 240.")


def ping_pong(servo):
    """Sweep back and forth between two user-entered angles until Ctrl+C."""
    try:
        a = ask_angle("first angle> ")
        b = ask_angle("second angle> ")
        raw = input(f"ms per sweep [{MOVE_TIME_MS}]> ").strip()
        sweep_ms = int(raw) if raw.isdigit() else MOVE_TIME_MS

        print(f"Looping {a} <-> {b} every {sweep_ms} ms. Press Ctrl+C to stop.")
        target = a
        while True:
            servo.move(target, time=sweep_ms)
            time.sleep(sweep_ms / 1000.0)
            print(f"  at {target} (servo reports {servo.get_physical_angle():.1f})")
            target = b if target == a else a
    except KeyboardInterrupt:
        print("\nLoop stopped.")


def connect(port):
    """Open the BusLinker port and set up the servo. Raises ServoDisconnected on failure."""
    # Close any old handle ourselves - LX16A.initialize() flushes the old port first, which
    # itself throws if that port vanished (BusLinker unplugged).
    if LX16A._controller is not None:
        try:
            LX16A._controller.close()
        except Exception:
            pass
        LX16A._controller = None

    print(f"Connecting to BusLinker on {port}...")
    try:
        LX16A.initialize(port)
        servo = LX16A(SERVO_ID)
    except (serial.SerialException, OSError) as exc:
        raise ServoDisconnected(f"Can't open {port} ({exc}) - is the BusLinker plugged in via USB?")
    except (ServoTimeoutError, ServoChecksumError):
        raise ServoDisconnected(
            f"No response from servo ID {SERVO_ID}. Check:\n"
            "  - the BusLinker has its own motor power connected (not just USB)\n"
            "  - the servo's 3-wire cable is plugged into one of the BusLinker's servo ports\n"
            f"  - SERVO_ID ({SERVO_ID}) matches the servo's actual bus ID (1 is the factory default)"
        )

    voltage = servo.get_vin() / 1000.0  # get_vin() reports millivolts
    temp = servo.get_temp()
    print(f"Connected to servo {SERVO_ID}. Input voltage: {voltage:.2f}V, temperature: {temp}C")
    if voltage < 6.0:
        print("Warning: that voltage looks low for an LX-16A (rated ~6-8.4V) - double check the power supply.")

    # A servo left in continuous-rotation (motor) mode ignores position commands.
    if servo.is_motor_mode():
        print("Servo was in motor mode - switching it back to servo (position) mode.")
        servo.servo_mode()
    return servo


def reconnect(port):
    """Keep offering to reconnect until it works. Returns the new servo, or None if the user quits."""
    while True:
        choice = input("Press Enter to reconnect, or 'q' to quit: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return None
        try:
            return connect(port)
        except ServoDisconnected as exc:
            print(exc)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else pick_port()

    try:
        servo = connect(port)
    except ServoDisconnected as exc:
        print(exc)
        sys.exit(1)

    print("Enter an angle from 0-240 to move the servo there, 'p' to loop between two angles, or 'q' to quit.")
    try:
        while True:
            raw = input("angle> ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                break
            if raw.lower() == "p":
                try:
                    ping_pong(servo)
                except DISCONNECT_ERRORS as exc:
                    servo = handle_disconnect(port, exc)
                    if servo is None:
                        break
                except Exception as exc:
                    print(f"Loop failed: {exc}")
                continue
            try:
                angle = float(raw)
            except ValueError:
                print("Not a number - try again.")
                continue
            try:
                # wait=False sends the move immediately. (wait=True only queues it on the
                # servo until move_start() is called, so the servo would never move.)
                servo.move(angle, time=MOVE_TIME_MS)
                time.sleep(MOVE_TIME_MS / 1000.0)
                print(f"Moved to {angle} degrees (servo reports {servo.get_physical_angle():.1f}).")
            except DISCONNECT_ERRORS as exc:
                servo = handle_disconnect(port, exc)
                if servo is None:
                    break
            except Exception as exc:
                print(f"Move failed: {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        # The servo may be gone, in which case there's no torque left to disable.
        if servo is not None:
            try:
                servo.disable_torque()
                print("\nTorque disabled, exiting.")
            except DISCONNECT_ERRORS:
                print("\nServo not reachable - couldn't disable torque, exiting.")
        else:
            print("\nExiting.")


def handle_disconnect(port, exc):
    """Explain a lost connection and try to recover. Returns the new servo, or None to quit."""
    print(f"\nLost connection to servo {SERVO_ID}: {exc}")
    if isinstance(exc, (serial.SerialException, OSError)):
        print("The BusLinker's USB connection dropped - plug it back in.")
    else:
        print("The servo stopped responding - check its cable and the BusLinker's motor power.")
    return reconnect(port)


if __name__ == "__main__":
    main()
