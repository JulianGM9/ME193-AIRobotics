"""Jog two daisy-chained LX-16A servos (bus IDs 1 and 8) from a small window.

Each servo gets a counterclockwise and a clockwise button: hold one down and
the servo keeps turning that way until you let go (or it reaches the end of
its range). Keyboard shortcuts do the same thing:
    servo 1: Left / Right arrow  (CCW / CW)
    servo 8: Up / Down arrow     (CCW / CW)

Setting your own zero: move a servo (or press Torque off and pose it by
hand), then press "Set zero here". From then on that position reads 0 deg and
angles are shown relative to it - so they can go negative. Zeros are saved
to servo_zeros.json next to this file and reloaded on the next run. The
servo's physical travel is still 0-240 raw degrees; the zero only changes
how it's numbered. (The LX-16A's own built-in offset only reaches +/-30 deg,
which is why this is done in software.)

The step slider sets how far a held button moves the servo per tick, i.e.
how fast it turns. If a servo drops off the bus its panel greys out; plug it
back in and press Reconnect.

Run with:
    python3 two_servo_gui.py                          # lists ports / prompts you to pick one
    python3 two_servo_gui.py /dev/cu.usbserial-110     # or pass the port directly
"""
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from pylx16a.lx16a import LX16A

from test1 import DISCONNECT_ERRORS, pick_port

SERVO_IDS = (1, 8)
JOG_INTERVAL_MS = 50  # how often a held button sends the next target - also each move's duration, so motion is continuous
READ_INTERVAL_MS = 250  # how often the angle readouts refresh
DEFAULT_STEP_DEG = 2.0  # degrees per jog tick -> 2 deg / 50 ms = 40 deg/s
# On these servos the raw angle increases clockwise looking at the horn (checked on the
# hardware). If your buttons turn the wrong way, flip this to -1.
CW_SIGN = 1
KEYS = {1: ("Left", "Right"), 8: ("Up", "Down")}  # servo ID -> (CCW key, CW key)
ZEROS_FILE = Path(__file__).with_name("servo_zeros.json")


def load_zeros():
    """{servo ID: raw angle that counts as 0}. Missing servos default to raw 0."""
    try:
        return {int(k): float(v) for k, v in json.loads(ZEROS_FILE.read_text()).items()}
    except (FileNotFoundError, ValueError):
        return {}


def save_zeros(zeros):
    ZEROS_FILE.write_text(json.dumps({str(k): v for k, v in sorted(zeros.items())}, indent=2) + "\n")


class ServoPanel:
    """Buttons, readout and jog state for one servo.

    Everything that talks to the servo happens in jog_tick()/refresh(), which App runs
    inside its disconnect handling - the button callbacks only record what was asked for.
    """

    def __init__(self, parent, servo_id, column, zeros):
        self.servo_id = servo_id
        self.zeros = zeros  # shared dict, saved whenever a zero changes
        self.servo = None
        self.target = None  # commanded raw angle we're jogging from, in degrees
        self.direction = 0  # +1 = increasing raw angle, -1 = decreasing, 0 = stopped
        self.pending_goto = None  # raw angle requested by "Go to 0"
        self.pending_zero = False  # "Set zero here" pressed, applied on the next read

        frame = ttk.LabelFrame(parent, text=f"Servo {servo_id}", padding=10)
        frame.grid(row=0, column=column, padx=8, pady=8, sticky="nsew")

        self.angle_var = tk.StringVar(value="--")
        ttk.Label(frame, textvariable=self.angle_var, font=("Menlo", 28)).grid(row=0, column=0, columnspan=3)
        self.detail_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.detail_var, foreground="gray").grid(row=1, column=0, columnspan=3)
        self.status_var = tk.StringVar(value="not connected")
        ttk.Label(frame, textvariable=self.status_var).grid(row=2, column=0, columnspan=3, pady=(0, 8))

        ccw_key, cw_key = KEYS[servo_id]
        self.ccw_btn = ttk.Button(frame, text=f"⟲ CCW  [{ccw_key}]")
        self.home_btn = ttk.Button(frame, text="Go to 0°", command=lambda: self.request_goto(self.zero))
        self.cw_btn = ttk.Button(frame, text=f"CW ⟳  [{cw_key}]")
        self.ccw_btn.grid(row=3, column=0, padx=2)
        self.home_btn.grid(row=3, column=1, padx=2)
        self.cw_btn.grid(row=3, column=2, padx=2)

        self.set_zero_btn = ttk.Button(frame, text="Set zero here", command=self.request_zero)
        self.clear_zero_btn = ttk.Button(frame, text="Clear zero", command=self.clear_zero)
        self.set_zero_btn.grid(row=4, column=0, columnspan=2, pady=(6, 0), sticky="ew", padx=2)
        self.clear_zero_btn.grid(row=4, column=2, pady=(6, 0), sticky="ew", padx=2)
        self.buttons = (self.ccw_btn, self.home_btn, self.cw_btn, self.set_zero_btn, self.clear_zero_btn)

        # Hold-to-jog: pressing starts motion, releasing stops it.
        self.ccw_btn.bind("<ButtonPress-1>", lambda e: self.start(-CW_SIGN))
        self.cw_btn.bind("<ButtonPress-1>", lambda e: self.start(CW_SIGN))
        for btn in (self.ccw_btn, self.cw_btn):
            btn.bind("<ButtonRelease-1>", lambda e: self.stop())

    @property
    def connected(self):
        return self.servo is not None

    @property
    def zero(self):
        return self.zeros.get(self.servo_id, 0.0)

    def attach(self, servo):
        self.servo = servo
        self.direction = 0
        self.pending_goto = None
        self.pending_zero = False
        self.target = self.clamp(servo.get_physical_angle())
        self.status_var.set("connected")
        for btn in self.buttons:
            btn.state(["!disabled"])

    def detach(self, reason):
        self.servo = None
        self.direction = 0
        self.angle_var.set("--")
        self.detail_var.set("")
        self.status_var.set(reason)
        for btn in self.buttons:
            btn.state(["disabled"])

    def clamp(self, angle):
        lo, hi = self.servo.get_angle_limits()  # the servo's stored limits (raw), cached at connect
        return max(lo, min(hi, angle))

    # --- button callbacks: just record the request ---

    def start(self, direction):
        if self.connected:
            self.direction = direction

    def stop(self):
        self.direction = 0

    def request_goto(self, raw_angle):
        if self.connected:
            self.pending_goto = raw_angle

    def request_zero(self):
        if self.connected:
            self.pending_zero = True

    def clear_zero(self):
        self.zeros.pop(self.servo_id, None)
        save_zeros(self.zeros)

    # --- bus access, called from App's loops ---

    def ensure_torque(self):
        """Torque off leaves the servo limp; the next commanded move picks up from wherever it was posed."""
        if not self.servo.is_torque_enabled():
            self.target = self.clamp(self.servo.get_physical_angle())
            self.servo.enable_torque()
            self.status_var.set("connected")

    def jog_tick(self, step):
        """Send any pending go-to, or advance the target one step in the held direction."""
        if not self.connected:
            return
        if self.pending_goto is not None:
            self.ensure_torque()
            self.target = self.clamp(self.pending_goto)
            self.pending_goto = None
            self.servo.move(self.target, time=600)
        elif self.direction != 0:
            self.ensure_torque()
            new_target = self.clamp(self.target + self.direction * step)
            if new_target != self.target:
                self.target = new_target
                self.servo.move(self.target, time=JOG_INTERVAL_MS)

    def refresh(self):
        if not self.connected:
            return
        raw = self.servo.get_physical_angle()
        if self.pending_zero:
            self.pending_zero = False
            self.zeros[self.servo_id] = round(raw, 2)
            save_zeros(self.zeros)
        lo, hi = self.servo.get_angle_limits()
        self.angle_var.set(f"{raw - self.zero:+7.1f}°")
        self.detail_var.set(
            f"raw {raw:.1f}° · zero at raw {self.zero:.1f}° · "
            f"range {lo - self.zero:+.0f}° to {hi - self.zero:+.0f}°"
        )


class App:
    def __init__(self, root, port):
        self.root = root
        self.port = port
        root.title("Servo jog - IDs " + " & ".join(map(str, SERVO_IDS)))

        body = ttk.Frame(root, padding=8)
        body.pack(fill="both", expand=True)
        zeros = load_zeros()
        self.panels = {sid: ServoPanel(body, sid, col, zeros) for col, sid in enumerate(SERVO_IDS)}

        controls = ttk.Frame(body)
        controls.grid(row=1, column=0, columnspan=len(SERVO_IDS), sticky="ew", padx=8)
        ttk.Label(controls, text="Step (deg/tick):").pack(side="left")
        self.step_var = tk.DoubleVar(value=DEFAULT_STEP_DEG)
        ttk.Scale(controls, from_=0.5, to=10.0, variable=self.step_var, length=160).pack(side="left", padx=4)
        self.step_label = ttk.Label(controls, width=5)
        self.step_label.pack(side="left")
        ttk.Button(controls, text="Torque off", command=self.torque_off).pack(side="right")
        ttk.Button(controls, text="Reconnect", command=self.connect).pack(side="right", padx=4)

        self.message_var = tk.StringVar()
        ttk.Label(body, textvariable=self.message_var, foreground="gray").grid(
            row=2, column=0, columnspan=len(SERVO_IDS), sticky="w", padx=8, pady=(6, 0))

        for sid, (ccw_key, cw_key) in KEYS.items():
            panel = self.panels[sid]
            for key, direction in ((ccw_key, -CW_SIGN), (cw_key, CW_SIGN)):
                root.bind(f"<KeyPress-{key}>", lambda e, p=panel, d=direction: p.start(d))
                root.bind(f"<KeyRelease-{key}>", lambda e, p=panel: p.stop())

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.connect()
        root.after(JOG_INTERVAL_MS, self.jog_loop)
        root.after(READ_INTERVAL_MS, self.read_loop)

    def connect(self):
        """(Re)open the BusLinker and attach whichever servos answer."""
        # Close the old handle ourselves - LX16A.initialize() flushes it first, which throws
        # if the BusLinker was unplugged.
        if LX16A._controller is not None:
            try:
                LX16A._controller.close()
            except Exception:
                pass
            LX16A._controller = None
        try:
            LX16A.initialize(self.port)
        except DISCONNECT_ERRORS as exc:
            for panel in self.panels.values():
                panel.detach("BusLinker not found")
            self.message_var.set(f"Can't open {self.port}: {exc}")
            return

        missing = []
        for sid, panel in self.panels.items():
            try:
                servo = LX16A(sid)
                if servo.is_motor_mode():
                    servo.servo_mode()
                panel.attach(servo)
            except DISCONNECT_ERRORS:
                panel.detach("no response")
                missing.append(str(sid))
        if missing:
            self.message_var.set(f"No reply from ID {', '.join(missing)} - check cables / motor power, then Reconnect.")
        else:
            self.message_var.set(f"Connected on {self.port}. Supply: {self.panels[SERVO_IDS[0]].servo.get_vin() / 1000:.2f} V")

    def on_disconnect(self, panel, exc):
        panel.detach("disconnected")
        self.message_var.set(f"Lost servo {panel.servo_id} ({exc}). Fix the connection, then press Reconnect.")

    def jog_loop(self):
        self.step_label.config(text=f"{self.step_var.get():.1f}")
        for panel in self.panels.values():
            try:
                panel.jog_tick(self.step_var.get())
            except DISCONNECT_ERRORS as exc:
                self.on_disconnect(panel, exc)
        self.root.after(JOG_INTERVAL_MS, self.jog_loop)

    def read_loop(self):
        # Moves never get a reply, so this read is where a dropped servo actually shows up.
        for panel in self.panels.values():
            try:
                panel.refresh()
            except DISCONNECT_ERRORS as exc:
                self.on_disconnect(panel, exc)
        self.root.after(READ_INTERVAL_MS, self.read_loop)

    def torque_off(self):
        """Let both servos go limp (e.g. to pose them by hand). Any move turns torque back on."""
        for panel in self.panels.values():
            if panel.connected:
                try:
                    panel.stop()
                    panel.servo.disable_torque()
                    panel.status_var.set("torque off (limp)")
                except DISCONNECT_ERRORS as exc:
                    self.on_disconnect(panel, exc)
        self.message_var.set("Torque off - pose by hand, Set zero if you like. Any jog turns torque back on.")

    def close(self):
        for panel in self.panels.values():
            if panel.connected:
                try:
                    panel.servo.disable_torque()
                except DISCONNECT_ERRORS:
                    pass
        self.root.destroy()


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else pick_port()
    root = tk.Tk()
    App(root, port)
    root.mainloop()


if __name__ == "__main__":
    main()
