"""Sends notifications on detection events. Telegram by default."""
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, cooldown_seconds: int = 30):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown_seconds = cooldown_seconds
        self._last_sent: Optional[float] = None

    def _in_cooldown(self) -> bool:
        if self._last_sent is None:
            return False
        return (time.time() - self._last_sent) < self.cooldown_seconds

    def send(
        self,
        message: str,
        face_count: int = 0,
        frame_number: int = 0,
        image_b64: Optional[str] = None,
    ) -> None:
        if self._in_cooldown():
            logger.debug("Notification suppressed (cooldown active)")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            resp = requests.post(
                url, json={"chat_id": self.chat_id, "text": message}, timeout=10
            )
            resp.raise_for_status()
            self._last_sent = time.time()
            logger.info("Telegram notification sent: %s", message)
        except requests.RequestException as e:
            logger.error("Failed to send Telegram notification: %s", e)


class DashboardNotifier:
    """
    Sends detection events to the Next.js dashboard's API route
    (POST /api/detections). Rate-limited by cooldown_seconds, same as
    TelegramNotifier.
    """

    def __init__(self, dashboard_url: str, api_key: str = "", cooldown_seconds: int = 30):
        # dashboard_url should be your deployed URL, e.g.
        # https://hik-dashboard.vercel.app
        self.endpoint = dashboard_url.rstrip("/") + "/api/detections"
        self.api_key = api_key
        self.cooldown_seconds = cooldown_seconds
        self._last_sent: Optional[float] = None

    def _in_cooldown(self) -> bool:
        if self._last_sent is None:
            return False
        return (time.time() - self._last_sent) < self.cooldown_seconds

    def send(
        self,
        message: str,
        face_count: int = 0,
        frame_number: int = 0,
        image_b64: Optional[str] = None,
    ) -> None:
        if self._in_cooldown():
            logger.debug("Dashboard notification suppressed (cooldown active)")
            return

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        payload = {
            "faceCount": face_count,
            "frameNumber": frame_number,
            "message": message,
        }
        if image_b64:
            payload["image"] = image_b64

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            self._last_sent = time.time()
            logger.info("Dashboard notified: %s", message)
        except requests.RequestException as e:
            logger.error("Failed to notify dashboard: %s", e)


class MultiNotifier:
    """Fans a single detection out to multiple notifiers."""

    def __init__(self, notifiers: list):
        self.notifiers = notifiers

    def send(
        self,
        message: str,
        face_count: int = 0,
        frame_number: int = 0,
        image_b64: Optional[str] = None,
    ) -> None:
        for notifier in self.notifiers:
            notifier.send(
                message,
                face_count=face_count,
                frame_number=frame_number,
                image_b64=image_b64,
            )
