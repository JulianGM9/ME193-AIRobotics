from pathlib import Path

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO

BROKER = "test.mosquitto.org"

GREEN_WEIGHTS_PATH = Path(__file__).parent / "runs" / "green_detector-3" / "weights" / "best.pt"
BLUE_WEIGHTS_PATH = Path(__file__).parent / "runs" / "blue_detector" / "weights" / "best.pt"

GREEN_TOPIC = "minifig_green"
BLUE_TOPIC = "minifig_blue"

CAMERA_INDEX = 0

# Minimum detection confidence (0-1) to count as "found". Edit this to
# tune sensitivity.
CONFIDENCE_THRESHOLD = 0.1


def to_coordinate(box, frame_width, frame_height):
    """Map a detection's center to (x, y) in [-1, 1].

    (0, 0) is the center of the frame. x increases to the right, y
    increases downward (standard image coordinate convention).
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    x = (cx - frame_width / 2) / (frame_width / 2)
    y = (cy - frame_height / 2) / (frame_height / 2)
    return x, y


def best_detection(results, confidence_threshold):
    """Return the highest-confidence box above threshold, or None."""
    best_box = None
    best_confidence = confidence_threshold
    boxes = results.boxes.xyxy.tolist()
    confidences = results.boxes.conf.tolist()
    for box, confidence in zip(boxes, confidences):
        if confidence > best_confidence:
            best_confidence = confidence
            best_box = box
    return best_box, best_confidence


def track_color(name, model, topic, draw_color, frame, client, confidence_threshold):
    """Run one color's model on the frame, publish its position if found,
    and draw its box on the frame. Returns nothing; frame is annotated
    in place."""
    frame_height, frame_width = frame.shape[:2]
    results = model.predict(frame, verbose=False)[0]
    box, confidence = best_detection(results, confidence_threshold)

    if box is not None:
        x, y = to_coordinate(box, frame_width, frame_height)
        payload = f"({x:.2f}, {y:.2f})"
        client.publish(topic, payload)
        print(f"[{name}] Published '{payload}' to '{topic}'  (confidence={confidence:.2f})")

        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
        cv2.putText(frame, name, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, draw_color, 2)
    else:
        print(f"[{name}] No minifig detected.")


def main():
    for weights_path in (GREEN_WEIGHTS_PATH, BLUE_WEIGHTS_PATH):
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Couldn't find trained weights at {weights_path}. "
                "Run train_yolo.py first, or update the weights path."
            )

    green_model = YOLO(str(GREEN_WEIGHTS_PATH))
    blue_model = YOLO(str(BLUE_WEIGHTS_PATH))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER)
    client.loop_start()

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    print(f"Tracking green ('{GREEN_TOPIC}') and blue ('{BLUE_TOPIC}') minifigs. "
          "Press 'q' in the preview window to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Failed to read from camera.")
                continue

            track_color("green", green_model, GREEN_TOPIC, (0, 255, 0),
                        frame, client, CONFIDENCE_THRESHOLD)
            track_color("blue", blue_model, BLUE_TOPIC, (255, 0, 0),
                        frame, client, CONFIDENCE_THRESHOLD)

            cv2.imshow("Minifig Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
