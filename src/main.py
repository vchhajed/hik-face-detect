import base64
import logging
import os
import sys
import time

import cv2

from .config import Config
from .face_detector import FaceDetector
from .hik_client import StreamUnavailable, build_client
from .notifier import DashboardNotifier, MultiNotifier, TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "face_detection_yunet.onnx"
)


def main() -> int:
    try:
        cfg = Config.load()
    except ValueError as e:
        logger.error("Config error: %s", e)
        return 1

    if not os.path.exists(MODEL_PATH):
        logger.error(
            "Face detection model not found at %s — see README setup step 2",
            MODEL_PATH,
        )
        return 1

    client = build_client(
        cfg.stream_mode,
        rtsp_url=cfg.rtsp_url,
        device_serial=cfg.device_serial,
        verification_code=cfg.verification_code,
        hcnetsdk_path=cfg.hcnetsdk_path,
    )
    detector = FaceDetector(MODEL_PATH, confidence_threshold=cfg.detection_confidence)

    notifiers = []
    if cfg.telegram_bot_token and cfg.telegram_chat_id:
        notifiers.append(
            TelegramNotifier(
                cfg.telegram_bot_token,
                cfg.telegram_chat_id,
                cooldown_seconds=cfg.notify_cooldown_seconds,
            )
        )
    if cfg.dashboard_url:
        notifiers.append(
            DashboardNotifier(
                cfg.dashboard_url,
                cfg.dashboard_api_key,
                cooldown_seconds=cfg.notify_cooldown_seconds,
            )
        )
    notifier = MultiNotifier(notifiers)
    logger.info(
        "Notifiers active: %s", ", ".join(n.__class__.__name__ for n in notifiers)
    )

    logger.info("Connecting (mode=%s)...", cfg.stream_mode)
    client.connect()
    logger.info("Connected. Starting detection loop.")

    frame_count = 0
    try:
        for frame in client.frames():
            frame_count += 1
            detections = detector.detect(frame)

            if detections:
                logger.info(
                    "Frame %d: %d face(s) detected", frame_count, len(detections)
                )
                image_b64 = None
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    image_b64 = base64.b64encode(buf).decode("ascii")
                notifier.send(
                    f"Face detected ({len(detections)}) at frame {frame_count}",
                    face_count=len(detections),
                    frame_number=frame_count,
                    image_b64=image_b64,
                )
            elif frame_count % 100 == 0:
                logger.debug("Frame %d: no faces", frame_count)

    except StreamUnavailable as e:
        logger.error("Stream lost: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
