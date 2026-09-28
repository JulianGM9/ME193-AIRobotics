import curses
import threading
import time

import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
TOPIC = "ME193/Julian"
PUBLISH_HZ = 10
STEP = 0.1

state_lock = threading.Lock()
speed = 0.0
turn = 0.0
running = True


def clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def publisher_loop(client):
    period = 1.0 / PUBLISH_HZ
    while running:
        with state_lock:
            payload = f"({speed:.2f}, {turn:.2f})"
        client.publish(TOPIC, payload)
        time.sleep(period)


def choose_role(client):
    while True:
        choice = input("Choose role - (g)oalie or (s)hooter: ").strip().lower()
        if choice in ("g", "goalie"):
            role = "goalie"
            break
        if choice in ("s", "shooter"):
            role = "shooter"
            break
        print("Please enter 'g' for goalie or 's' for shooter.")
    client.publish(TOPIC, role)
    print(f"Sent role '{role}' on '{TOPIC}'.")
    return role


def run_controller(stdscr, client, role):
    global speed, turn

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, f"Test controller  |  role: {role}  |  broker: {BROKER}  |  topic: {TOPIC}")
        with state_lock:
            current_speed, current_turn = speed, turn
        stdscr.addstr(2, 0, f"Speed: {current_speed:+.2f}   Turn: {current_turn:+.2f}")
        stdscr.addstr(3, 0, f"(broadcasting at {PUBLISH_HZ} Hz)")
        stdscr.addstr(5, 0, "Up/Down: speed    Left/Right: turn    Space: stop")
        stdscr.addstr(6, 0, "w: send WIN    l: send LOSE    q: quit")
        stdscr.refresh()

        ch = stdscr.getch()
        if ch == -1:
            continue
        elif ch == curses.KEY_UP:
            with state_lock:
                speed = clamp(speed + STEP)
        elif ch == curses.KEY_DOWN:
            with state_lock:
                speed = clamp(speed - STEP)
        elif ch == curses.KEY_RIGHT:
            with state_lock:
                turn = clamp(turn + STEP)
        elif ch == curses.KEY_LEFT:
            with state_lock:
                turn = clamp(turn - STEP)
        elif ch == ord(" "):
            with state_lock:
                speed = 0.0
                turn = 0.0
        elif ch in (ord("w"), ord("W")):
            client.publish(TOPIC, "win")
        elif ch in (ord("l"), ord("L")):
            client.publish(TOPIC, "lose")
        elif ch in (ord("q"), ord("Q")):
            break


def main():
    global running

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER)
    client.loop_start()

    role = choose_role(client)

    publisher_thread = threading.Thread(target=publisher_loop, args=(client,), daemon=True)
    publisher_thread.start()

    try:
        curses.wrapper(run_controller, client, role)
    finally:
        running = False
        publisher_thread.join(timeout=1)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
