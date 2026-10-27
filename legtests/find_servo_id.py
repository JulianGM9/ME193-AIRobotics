"""Find an LX-16A servo's current bus ID by scanning, and optionally set
it to a new one.

An LX-16A simply never responds to a command addressed to the wrong ID,
so there's no "ask an unknown servo what its ID is" command - this just
tries connecting at every legal ID (0-253) and reports which one(s)
actually answer. Only run this with ONE servo connected to the bus at a
time - if two servos happened to share an ID you couldn't tell them apart
here anyway (LX-16A IDs are meant to be unique per bus).

Run with:
    python3 find_servo_id.py                 # scan only, report the ID found
    python3 find_servo_id.py --set-id 1       # scan, then set the found servo's ID to 1
"""
import argparse
import sys

import serial.tools.list_ports
from pylx16a.lx16a import LX16A, ServoTimeoutError

MIN_ID, MAX_ID = 0, 253  # the full legal range per the protocol (pylx16a's own set_id() enforces this)
# A bit more forgiving than pylx16a's 0.02s default - this is a one-time
# diagnostic, not a tight control loop, so trading some speed for a more
# reliable scan is worth it.
SCAN_TIMEOUT_SECONDS = 0.1


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


def scan_for_servo():
    found = []
    for candidate_id in range(MIN_ID, MAX_ID + 1):
        if candidate_id % 25 == 0:
            print(f"  ...checked up to ID {candidate_id}")
        try:
            found.append(LX16A(candidate_id, disable_torque=True))
        except ServoTimeoutError:
            pass
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-id", type=int, default=None,
                         help="if exactly one servo is found, set its ID to this value")
    args = parser.parse_args()

    port = pick_port()
    print(f"Connecting to BusLinker on {port}...")
    LX16A.initialize(port, timeout=SCAN_TIMEOUT_SECONDS)

    print(f"Scanning IDs {MIN_ID}-{MAX_ID} (make sure only one servo is on the bus)...")
    found = scan_for_servo()

    if not found:
        print("No servo responded anywhere in the legal ID range. Check wiring/power "
              "(see test1.py's setup notes) rather than assuming the ID is the problem.")
        sys.exit(1)

    if len(found) > 1:
        ids = [s.get_id() for s in found]
        print(f"Found {len(found)} servos responding (IDs {ids}) - this script assumes "
              "just one servo on the bus at a time. Disconnect all but one and try again.")
        sys.exit(1)

    servo = found[0]
    current_id = servo.get_id()
    print(f"Found it: current servo ID is {current_id}")

    if args.set_id is not None:
        servo.set_id(args.set_id)
        print(f"ID changed to {args.set_id}. Update SERVO_ID in test1.py to match if you use it there.")


if __name__ == "__main__":
    main()
