import math
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

import legoeducation as le

# --- Hardware ---
# Identifies this specific hub by its Connection Card, rather than
# connecting to the first Double Motor found.
CARD_COLOR = le.LEGO_COLOR_YELLOW
CARD_SERIAL = "1131"

# --- State space: discretized yaw error ---
# Yaw error is binned into fixed-width buckets covering a limited range;
# anything beyond the range clips to the outermost bucket. Narrower bins
# give finer control but need more episodes to learn; wider bins learn
# faster but steer more coarsely.
YAW_BIN_WIDTH = 10  # degrees per bucket
YAW_RANGE = 60  # bucket edges cover [-YAW_RANGE, YAW_RANGE]
NUM_STATES = (2 * YAW_RANGE) // YAW_BIN_WIDTH + 1


def yaw_to_state(yaw_degrees):
    """Map a yaw reading (degrees, 0 = straight) to a discrete state index."""
    clipped = max(-YAW_RANGE, min(YAW_RANGE, yaw_degrees))
    bucket = round((clipped + YAW_RANGE) / YAW_BIN_WIDTH)
    return int(bucket)


# --- Action space: discrete motor power presets (left, right) ---
ACTIONS = [
    (-30, 30),  # hard left
    (-10, 30),  # soft left
    (30, 30),  # straight
    (30, -10),  # soft right
    (30, -30),  # hard right
]
ACTION_LABELS = ["hard L", "soft L", "straight", "soft R", "hard R"]
NUM_ACTIONS = len(ACTIONS)


# --- Reward ---
def reward_for_yaw(yaw_degrees):
    return -abs(yaw_degrees)


# --- Hyperparameters ---
ALPHA = 0.2  # learning rate: how much each update shifts a Q-value
GAMMA = 0.9  # discount factor: how much future reward matters
EPSILON_START = 1.0  # exploration rate at the start of training
EPSILON_MIN = 0.05  # exploration rate floor after decay
EPSILON_DECAY = 0.995  # multiply epsilon by this after each episode

STEP_DURATION = 0.3  # seconds to hold each action before measuring its effect
DEFAULT_STEPS_PER_EPISODE = 20
DEFAULT_TEST_STEPS = 20

# --- Q-table ---
# Q[state, action] = expected future reward. Loaded from disk if a
# previous session already trained one, so training can resume across
# separate runs instead of starting from scratch every time.
Q_TABLE_PATH = Path(__file__).parent / "q_table.npy"
if Q_TABLE_PATH.exists():
    q_table = np.load(Q_TABLE_PATH)
else:
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))


def choose_action(state, epsilon):
    """Epsilon-greedy: a random action with probability epsilon, otherwise
    the best-known action for this state."""
    if np.random.random() < epsilon:
        return np.random.randint(NUM_ACTIONS)
    return int(np.argmax(q_table[state]))


# --- Worker thread / UI shared state ---
# The UI thread must never block on hardware I/O, so all motor/sensor
# access happens on a background thread. `shared` is what the UI polls to
# redraw; `commands` is how the UI tells the worker what to do next.
lock = threading.Lock()
shared = {
    "yaw": 0.0,
    "episode": 0,
    "epsilon": EPSILON_START,
    "status": "connecting",
    "last_reward": None,
}
commands = queue.Queue()
stop_event = threading.Event()  # set by the "Stop" button to end a run early


def set_status(text):
    with lock:
        shared["status"] = text


def run_steps(motor, steps, epsilon, learn):
    """Run up to `steps` actions (stopping early if stop_event is set).

    If learn=True, this updates q_table (training). If learn=False, it
    acts purely greedily (epsilon=0) and never writes to q_table - an
    independent evaluation of the current policy, not more training.
    """
    state = yaw_to_state(motor.imu_device.yaw)
    total_reward = 0.0

    for _ in range(steps):
        if stop_event.is_set():
            break

        action = choose_action(state, epsilon if learn else 0.0)
        left, right = ACTIONS[action]
        motor.movement_move_tank(left, right, blocking=False)
        time.sleep(STEP_DURATION)

        yaw = motor.imu_device.yaw
        with lock:
            shared["yaw"] = yaw

        next_state = yaw_to_state(yaw)
        reward = reward_for_yaw(yaw)
        total_reward += reward

        if learn:
            best_next = np.max(q_table[next_state])
            q_table[state, action] += ALPHA * (reward + GAMMA * best_next - q_table[state, action])

        state = next_state

    stop_event.clear()
    motor.movement_stop()
    return total_reward


def worker_main():
    motor = le.DoubleMotor()
    set_status("connecting")
    motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    with lock:
        shared["yaw"] = motor.imu_device.yaw
    set_status("idle - zero yaw to begin")

    epsilon = EPSILON_START
    with lock:
        shared["epsilon"] = epsilon

    try:
        while True:
            command = commands.get()
            name = command[0]

            if name == "quit":
                break

            elif name == "zero_yaw":
                motor.imu_reset_yaw_axis(0)
                with lock:
                    shared["yaw"] = 0.0
                set_status("idle")

            elif name == "episode":
                steps = command[1]
                set_status("training")
                total_reward = run_steps(motor, steps, epsilon, learn=True)
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
                np.save(Q_TABLE_PATH, q_table)
                with lock:
                    shared["episode"] += 1
                    shared["epsilon"] = epsilon
                    shared["last_reward"] = total_reward
                set_status("idle")

            elif name == "test":
                steps = command[1]
                set_status("testing (not learning)")
                total_reward = run_steps(motor, steps, epsilon, learn=False)
                with lock:
                    shared["last_reward"] = total_reward
                set_status("idle")
    finally:
        motor.movement_stop()
        motor.disconnect()
        np.save(Q_TABLE_PATH, q_table)


class App:
    def __init__(self, root):
        self.root = root
        root.title("MLwalker Q-learning trainer")

        controls = ttk.Frame(root, padding=10)
        controls.grid(row=0, column=0, sticky="n")

        ttk.Button(controls, text="Zero Yaw (set 0°)", command=self.zero_yaw).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(controls, text="Steps/episode:").grid(row=1, column=0, sticky="w")
        self.episode_steps_var = tk.StringVar(value=str(DEFAULT_STEPS_PER_EPISODE))
        ttk.Entry(controls, textvariable=self.episode_steps_var, width=6).grid(row=1, column=1)
        ttk.Button(controls, text="New Episode (train)", command=self.start_episode).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(controls, text="Test steps:").grid(row=3, column=0, sticky="w")
        self.test_steps_var = tk.StringVar(value=str(DEFAULT_TEST_STEPS))
        ttk.Entry(controls, textvariable=self.test_steps_var, width=6).grid(row=3, column=1)
        ttk.Button(controls, text="Independent Test Run", command=self.start_test).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Button(controls, text="Stop current run", command=self.stop_run).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.status_var = tk.StringVar(value="connecting...")
        ttk.Label(controls, textvariable=self.status_var, justify="left").grid(
            row=6, column=0, columnspan=2, sticky="w")

        self.canvas = tk.Canvas(root, width=220, height=260, bg="white", highlightthickness=1)
        self.canvas.grid(row=0, column=1, padx=10, pady=10)

        table_frame = ttk.Frame(root, padding=10)
        table_frame.grid(row=0, column=2, sticky="n")
        ttk.Label(table_frame, text="Q-table", font=("", 11, "bold")).grid(
            row=0, column=0, columnspan=NUM_ACTIONS + 1)
        for a, label in enumerate(ACTION_LABELS):
            ttk.Label(table_frame, text=label, font=("", 8)).grid(row=1, column=a + 1)

        self.q_cells = []
        for s in range(NUM_STATES):
            yaw_center = s * YAW_BIN_WIDTH - YAW_RANGE
            ttk.Label(table_frame, text=f"{yaw_center:+d}°", font=("", 8)).grid(row=s + 2, column=0)
            row_cells = []
            for a in range(NUM_ACTIONS):
                cell = tk.Label(table_frame, text="0.0", width=7, relief="groove")
                cell.grid(row=s + 2, column=a + 1)
                row_cells.append(cell)
            self.q_cells.append(row_cells)

        self.update_ui()

    def zero_yaw(self):
        commands.put(("zero_yaw",))

    def start_episode(self):
        try:
            steps = int(self.episode_steps_var.get())
        except ValueError:
            steps = DEFAULT_STEPS_PER_EPISODE
        commands.put(("episode", steps))

    def start_test(self):
        try:
            steps = int(self.test_steps_var.get())
        except ValueError:
            steps = DEFAULT_TEST_STEPS
        commands.put(("test", steps))

    def stop_run(self):
        stop_event.set()

    def update_ui(self):
        with lock:
            yaw = shared["yaw"]
            status = shared["status"]
            episode = shared["episode"]
            epsilon = shared["epsilon"]
            last_reward = shared["last_reward"]

        reward_text = f"{last_reward:.1f}" if last_reward is not None else "-"
        self.status_var.set(
            f"Status: {status}\n"
            f"Episode: {episode}   Epsilon: {epsilon:.3f}\n"
            f"Last run's total reward: {reward_text}"
        )

        self.draw_compass(yaw)

        current_state = yaw_to_state(yaw)
        table_snapshot = q_table.copy()
        best_actions = np.argmax(table_snapshot, axis=1)
        for s in range(NUM_STATES):
            for a in range(NUM_ACTIONS):
                cell = self.q_cells[s][a]
                cell.configure(text=f"{table_snapshot[s, a]:.1f}")
                if s == current_state and a == best_actions[s]:
                    cell.configure(background="#8fe08f")
                elif s == current_state:
                    cell.configure(background="#cdd6ff")
                elif a == best_actions[s]:
                    cell.configure(background="#e7ffe7")
                else:
                    cell.configure(background="white")

        self.root.after(150, self.update_ui)

    def draw_compass(self, yaw):
        c = self.canvas
        c.delete("all")
        cx, cy, r = 110, 120, 90
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#999999")

        # Target direction is always "up" (0 degrees), by definition of
        # how yaw is zeroed. Current heading arrow rotates by the live
        # yaw error relative to that fixed reference.
        self._draw_arrow(cx, cy, r, 0, "#2a9d2a")
        self._draw_arrow(cx, cy, r, yaw, "#d1495b")

        c.create_text(cx, cy + r + 15, text="green = target   red = current", font=("", 9))
        c.create_text(cx, cy + r + 35, text=f"Yaw error: {yaw:+.1f}°", font=("", 13, "bold"))

    def _draw_arrow(self, cx, cy, r, yaw_degrees, color):
        angle = math.radians(yaw_degrees)
        x = cx + r * math.sin(angle)
        y = cy - r * math.cos(angle)
        self.canvas.create_line(cx, cy, x, y, fill=color, width=3, arrow=tk.LAST)


def main():
    worker_thread = threading.Thread(target=worker_main, daemon=True)
    worker_thread.start()

    root = tk.Tk()
    app = App(root)

    def on_close():
        stop_event.set()
        commands.put(("quit",))
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
