import cv2.aruco as aruco

from vision.level_selector import LevelSelector  # re-exported for convenience/back-compat

DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)


def create_detector():
    params = aruco.DetectorParameters()
    return aruco.ArucoDetector(DICTIONARY, params)


def detect_tags(frame_bgr, detector):
    """Returns a list of (tag_id, corners) for every AprilTag found in the
    frame. corners is a (4, 2) array of pixel coordinates, in order
    top-left, top-right, bottom-right, bottom-left."""
    corners_list, ids, _ = detector.detectMarkers(frame_bgr)
    if ids is None:
        return []
    return [(int(tag_id), corners[0]) for tag_id, corners in zip(ids, corners_list)]
