import threading

import cv2

MAX_CONSECUTIVE_FAILURES = 30


class Camera:
    """Continuously captures frames on a background thread, keeping only
    the single latest one. Consumers call get_latest_frame() and naturally
    drop any frame they didn't have time to process - there's no queue to
    fall behind on, which is what lets pose inference always work on the
    freshest frame instead of catching up on stale ones.
    """

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._stop = threading.Event()
        self._thread = None
        self.opened = False

    def open(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # isOpened() can lie (notably on macOS); the real test is whether a
        # frame actually arrives.
        for _ in range(10):
            ok, frame = self._cap.read()
            if ok:
                self._latest_frame = frame
                self.opened = True
                return True
        self._cap.release()
        return False

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                failures += 1
                if failures > MAX_CONSECUTIVE_FAILURES:
                    self.opened = False
                    return
                continue
            failures = 0
            with self._lock:
                self._latest_frame = frame

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
