"""Loads and validates configuration from .env."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    stream_mode: str
    rtsp_url: str
    device_serial: str
    verification_code: str
    hcnetsdk_path: str
    detection_confidence: float
    notify_cooldown_seconds: int
    telegram_bot_token: str
    telegram_chat_id: str
    dashboard_url: str
    dashboard_api_key: str

    @classmethod
    def load(cls) -> "Config":
        stream_mode = os.getenv("STREAM_MODE", "rtsp").strip().lower()

        cfg = cls(
            stream_mode=stream_mode,
            rtsp_url=os.getenv("RTSP_URL", ""),
            device_serial=os.getenv("DEVICE_SERIAL", ""),
            verification_code=os.getenv("VERIFICATION_CODE", ""),
            hcnetsdk_path=os.getenv("HCNETSDK_PATH", ""),
            detection_confidence=float(os.getenv("DETECTION_CONFIDENCE", "0.8")),
            notify_cooldown_seconds=int(os.getenv("NOTIFY_COOLDOWN_SECONDS", "30")),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            dashboard_url=os.getenv("DASHBOARD_URL", ""),
            dashboard_api_key=os.getenv("DASHBOARD_API_KEY", ""),
        )

        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if self.stream_mode not in ("rtsp", "cloud"):
            raise ValueError("STREAM_MODE must be 'rtsp' or 'cloud'")

        if self.stream_mode == "rtsp" and not self.rtsp_url:
            raise ValueError("RTSP_URL is required when STREAM_MODE=rtsp")

        if self.stream_mode == "cloud":
            missing = [
                name
                for name, val in (
                    ("DEVICE_SERIAL", self.device_serial),
                    ("VERIFICATION_CODE", self.verification_code),
                    ("HCNETSDK_PATH", self.hcnetsdk_path),
                )
                if not val
            ]
            if missing:
                raise ValueError(
                    f"STREAM_MODE=cloud requires: {', '.join(missing)}"
                )

        has_telegram = bool(self.telegram_bot_token and self.telegram_chat_id)
        has_dashboard = bool(self.dashboard_url)

        if not has_telegram and not has_dashboard:
            raise ValueError(
                "Configure at least one notifier: either "
                "TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID, or DASHBOARD_URL"
            )
