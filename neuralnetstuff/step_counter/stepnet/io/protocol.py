"""Wire format between the board and the laptop (PLAN.md D2, D4).

Must match board/python/main.py and board/sketch/sketch.ino.

MQTT topics (all JSON):
  .../imu     board -> laptop  {"run": str, "t": board publish time, "b": batch string}
              published every 0.25 s (~25 samples): the public broker drops messages
              above ~8/s per client (measured 2026-11-02)
  .../status  board -> laptop  {"type": "status", "run", "online", "mcu", "samples", "gaps",
                                "rate_hz", "uptime_s", "log"}
                               {"type": "pong", "run", "id", "t"}
  .../cmd     laptop -> board  {"type": "ping", "id": int, "t": laptop send time}

A batch string is samples separated by ';', each  seq,t_ms,ax,ay,az,gx,gy,gz  as raw ints.
"""
import os

import numpy as np

BROKER = "broker.emqx.io"
PORT = 1883
# STEPNET_BASE_TOPIC lets tests (fake_board.py + any script) use a private topic while the
# real board keeps streaming on the normal one.
BASE_TOPIC = os.environ.get("STEPNET_BASE_TOPIC", "ME193/Julian/StepCounter")
TOPIC_IMU = BASE_TOPIC + "/imu"
TOPIC_STATUS = BASE_TOPIC + "/status"
TOPIC_CMD = BASE_TOPIC + "/cmd"

SAMPLE_RATE_HZ = 100.0
ACCEL_LSB_PER_G = 2048.0  # +/-16 g range
GYRO_LSB_PER_DPS = 16.4  # +/-2000 deg/s range

COLUMNS = ("seq", "t_ms", "ax", "ay", "az", "gx", "gy", "gz")


def parse_batch(batch: str) -> np.ndarray:
    """Batch string -> int64 array of shape (n_samples, 8) with COLUMNS. Raises ValueError if malformed."""
    rows = [s.split(",") for s in batch.split(";") if s]
    if not rows or any(len(r) != len(COLUMNS) for r in rows):
        raise ValueError(f"malformed batch: {batch[:80]!r}")
    return np.array(rows, dtype=np.int64)


def format_batch(samples: np.ndarray) -> str:
    """Inverse of parse_batch (used by the fake board and tests)."""
    return ";".join(",".join(str(int(v)) for v in row) for row in samples)
