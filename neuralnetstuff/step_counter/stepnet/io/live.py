"""A live sample buffer fed by BoardLink in a background thread.

Scripts that spend most of their time waiting for the user (calibrate_accel.py, record.py)
can't drain BoardLink's queue themselves, so LiveStream does it and keeps every sample
received since start(). Samples are rows of (seq, t_ms, ax, ay, az, gx, gy, gz) raw ints.
"""
import queue
import threading
import time

import numpy as np

from stepnet.io.mqtt_link import BoardLink
from stepnet.io.protocol import SAMPLE_RATE_HZ
from stepnet.io.stream import SeqTracker


class LiveStream:
    def __init__(self, link=None):
        self.link = link or BoardLink()
        self.tracker = SeqTracker()
        self.run = None
        self.board_status = {}
        self.messages = []  # link/connection notes, for the caller to print
        self._chunks = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True)

    def start(self):
        self.link.start()
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.link.stop()

    def _drain(self):
        while not self._stop.is_set():
            try:
                event = self.link.events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event[0] == "imu":
                _, run, _, samples = event
                with self._lock:
                    self.tracker.update(run, samples[:, 0])
                    self.run = run
                    self._chunks.append(samples)
            elif event[0] == "status":
                self.board_status = event[1]
            elif event[0] == "link":
                self.messages.append(event[2])

    def samples(self) -> np.ndarray:
        """Every sample received so far, shape (n, 8)."""
        with self._lock:
            if len(self._chunks) > 1:
                self._chunks = [np.concatenate(self._chunks)]
            return self._chunks[0].copy() if self._chunks else np.empty((0, 8), dtype=np.int64)

    def latest_seq(self):
        with self._lock:
            return int(self._chunks[-1][-1, 0]) if self._chunks else None

    def since(self, seq) -> np.ndarray:
        """Samples with seq >= the given seq."""
        s = self.samples()
        return s[s[:, 0] >= seq]

    def wait_for_data(self, timeout_s=15.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.latest_seq() is not None:
                return True
            time.sleep(0.1)
        return False

    def collect(self, seconds, stall_timeout_s=5.0) -> np.ndarray:
        """Block for `seconds` (on the sensor's clock) and return the samples recorded meanwhile.

        Raises TimeoutError if no new sample arrives for stall_timeout_s.
        """
        start = self.latest_seq()
        if start is None:
            raise TimeoutError("no IMU data yet - call wait_for_data() first")
        n = int(round(seconds * SAMPLE_RATE_HZ))
        last_seen, last_change = start, time.monotonic()
        while True:
            got = self.since(start + 1)
            if len(got) and got[-1, 0] - start >= n:
                return got[got[:, 0] <= start + n]
            latest = self.latest_seq()
            if latest != last_seen:
                last_seen, last_change = latest, time.monotonic()
            elif time.monotonic() - last_change > stall_timeout_s:
                raise TimeoutError(f"no IMU data for {stall_timeout_s:.0f} s")
            time.sleep(0.05)
