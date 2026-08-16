"""Runs FaceDetector over a local video file (e.g. a downloaded DVR segment)."""
import base64
import logging

import cv2

from .face_detector import FaceDetector
from .notifier import MultiNotifier

logger = logging.getLogger(__name__)


def process_file(
    video_path: str,
    detector: FaceDetector,
    notifier: MultiNotifier,
    sample_every_n_frames: int = 5,
    source_label: str = "",
) -> int:
    """
    Runs detection over every Nth frame of video_path, notifying on each
    frame with a detection. Returns the total number of frames with at
    least one detection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frame_count = 0
    hit_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % sample_every_n_frames != 0:
                continue

            detections = detector.detect(frame)
            if not detections:
                continue

            hit_count += 1
            logger.info(
                "%s frame %d: %d face(s) detected",
                source_label or video_path,
                frame_count,
                len(detections),
            )
            image_b64 = None
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                image_b64 = base64.b64encode(buf).decode("ascii")

            message = f"Face detected ({len(detections)}) in {source_label or video_path}"
            notifier.send(
                message,
                face_count=len(detections),
                frame_number=frame_count,
                image_b64=image_b64,
            )
    finally:
        cap.release()

    logger.info(
        "%s: processed %d frame(s) sampled, %d with detections",
        source_label or video_path,
        frame_count // sample_every_n_frames,
        hit_count,
    )
    return hit_count
