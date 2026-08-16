"""
CLI: search a DVR time range, download each matching recording, run face
detection over it, notify on detections, then delete the local copy.

Usage:
  python -m src.dvr_batch \\
    --host 192.168.1.5 --username admin --password 12345abc \\
    --track 101 \\
    --start "2026-07-28T00:00:00" --end "2026-07-28T23:59:59" \\
    --dry-run   # remove this flag once the search results look right
"""
import argparse
import logging
import os
import sys
import tempfile

from .config import Config
from .dvr_client import DvrClient, DvrError
from .face_detector import FaceDetector
from .notifier import DashboardNotifier, MultiNotifier, TelegramNotifier
from .process_file import process_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "face_detection_yunet.onnx"
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process DVR recordings for a time range")
    p.add_argument("--host", required=True, help="DVR IP or hostname")
    p.add_argument("--port", type=int, default=80, help="DVR HTTP port (default 80)")
    p.add_argument("--username", required=True)
    p.add_argument("--password", required=True)
    p.add_argument(
        "--track",
        type=int,
        required=True,
        help="Track ID: channel N main stream = N*100+1, sub = N*100+2",
    )
    p.add_argument("--start", required=True, help="ISO 8601 start time, e.g. 2026-07-28T00:00:00")
    p.add_argument("--end", required=True, help="ISO 8601 end time")
    p.add_argument(
        "--sample-every-n-frames",
        type=int,
        default=5,
        help="Run detection on every Nth frame (default 5)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only search and print matches, don't download/process anything",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        cfg = Config.load()
    except ValueError as e:
        logger.error("Config error: %s", e)
        return 1

    client = DvrClient(host=args.host, username=args.username, password=args.password, port=args.port)

    try:
        matches = client.search_recordings(args.track, args.start, args.end)
    except DvrError as e:
        logger.error("Search failed: %s", e)
        return 1

    if not matches:
        logger.info(
            "No recordings found for track %d between %s and %s "
            "(if you expected results, double check --track against the DVR's actual channel numbering)",
            args.track,
            args.start,
            args.end,
        )
        return 0

    for m in matches:
        logger.info("Match: %s -> %s (%s)", m.start_time, m.end_time, m.playback_uri)

    if args.dry_run:
        logger.info("Dry run: %d match(es) found, not downloading", len(matches))
        return 0

    if not os.path.exists(MODEL_PATH):
        logger.error(
            "Face detection model not found at %s — see README setup step 2",
            MODEL_PATH,
        )
        return 1

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

    total_hits = 0
    with tempfile.TemporaryDirectory(prefix="dvr_batch_") as tmpdir:
        for i, m in enumerate(matches, start=1):
            dest_path = os.path.join(tmpdir, f"segment_{i}.mp4")
            source_label = f"{args.host} track {args.track} [{m.start_time} - {m.end_time}]"
            try:
                client.download_recording(m, dest_path)
            except DvrError as e:
                logger.error("Download failed for %s: %s", m.playback_uri, e)
                continue

            try:
                hits = process_file(
                    dest_path,
                    detector,
                    notifier,
                    sample_every_n_frames=args.sample_every_n_frames,
                    source_label=source_label,
                )
                total_hits += hits
            except RuntimeError as e:
                logger.error("Processing failed for %s: %s", dest_path, e)
            finally:
                if os.path.exists(dest_path):
                    os.remove(dest_path)

    logger.info("Done. %d segment(s) processed, %d with face detections", len(matches), total_hits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
