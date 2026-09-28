import curses
import queue
import textwrap
from datetime import datetime

import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
TOPIC = "ME193"

incoming_messages = queue.Queue()


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(TOPIC)


def on_message(client, userdata, message):
    incoming_messages.put(message.payload.decode())


def run_chat(stdscr, client):
    curses.curs_set(1)
    height, width = stdscr.getmaxyx()

    chat_height = height - 3
    chat_win = curses.newwin(chat_height, width, 0, 0)
    chat_win.scrollok(True)
    chat_win.idlok(True)

    input_win = curses.newwin(3, width, chat_height, 0)
    input_win.nodelay(True)

    def log(text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        for line in textwrap.wrap(f"[{timestamp}] {text}", width - 2) or [""]:
            chat_win.addstr(f"{line}\n")
        chat_win.refresh()

    def redraw_input(buffer):
        input_win.erase()
        input_win.box()
        input_win.addstr(0, 2, f" #{TOPIC} (Enter to send, Ctrl+C to quit) ")
        input_win.addstr(1, 2, buffer[: width - 5])
        input_win.refresh()

    log(f"Connected to {BROKER}, subscribed to '{TOPIC}'.")
    buffer = ""
    redraw_input(buffer)

    while True:
        try:
            while True:
                log(incoming_messages.get_nowait())
        except queue.Empty:
            pass

        try:
            ch = input_win.getch(1, min(2 + len(buffer), width - 3))
        except curses.error:
            ch = -1

        if ch == -1:
            curses.napms(50)
            continue
        elif ch in (curses.KEY_ENTER, 10, 13):
            text = buffer.strip()
            if text:
                client.publish(TOPIC, text)
                log(f"me: {text}")
            buffer = ""
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            buffer = buffer[:-1]
        elif 32 <= ch <= 126 and len(buffer) < width - 6:
            buffer += chr(ch)

        redraw_input(buffer)


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER)
    client.loop_start()

    try:
        curses.wrapper(run_chat, client)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
