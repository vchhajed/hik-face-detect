"""
Camera stream client.

Two backends:

  RtspClient   - opens an RTSP URL directly with OpenCV. Works whenever the
                 machine running this code can reach the camera's IP
                 (same LAN, or via VPN/Tailscale). This is the reliable,
                 well-documented path.

  CloudClient  - logs in via Hikvision's cloud P2P using the device serial +
                 verification code (no IP needed, works from anywhere).
                 This wraps the `hikvision-sdk` package's login, which is
                 primarily documented for LAN IP login.

                 !! VERIFY BEFORE RELYING ON THIS !!
                 The exact call signature for a *cloud* (serial-based, no
                 IP) login vs a *LAN* (IP-based) login differs in the
                 underlying HCNetSDK. Once you've installed `hikvision-sdk`
                 and can inspect its source (or Hikvision's official SDK
                 docs from open.hikvision.com), confirm whether `HCNetSDK`
                 exposes a serial+code login path, or whether you need to
                 fall back to calling `NET_DVR_Login_V40` directly via
                 ctypes with an ISUP/cloud LOGIN_INFO struct. The method
                 below (`_cloud_login`) is written defensively and will
                 raise a clear NotImplementedError if the wrapper doesn't
                 support it, rather than silently failing.
"""
import time
from typing import Iterator, Optional

import cv2
import numpy as np


class StreamUnavailable(Exception):
    """Raised when a frame can't be read and retries are exhausted."""


class RtspClient:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self._cap: Optional[cv2.VideoCapture] = None

    def connect(self) -> None:
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise StreamUnavailable(f"Could not open RTSP stream: {self.rtsp_url}")

    def frames(self, max_consecutive_failures: int = 30) -> Iterator[np.ndarray]:
        if self._cap is None:
            self.connect()

        failures = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                failures += 1
                if failures >= max_consecutive_failures:
                    raise StreamUnavailable(
                        "Too many consecutive frame read failures — "
                        "reconnecting may be needed"
                    )
                time.sleep(0.1)
                continue
            failures = 0
            yield frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


class CloudClient:
    """
    Cloud P2P client using device serial + verification code.

    See module docstring — the cloud login path needs to be confirmed
    against the actual hikvision-sdk API surface once installed.
    """

    def __init__(self, device_serial: str, verification_code: str, hcnetsdk_path: str):
        self.device_serial = device_serial
        self.verification_code = verification_code
        self.hcnetsdk_path = hcnetsdk_path
        self._device = None
        self._sdk = None

    def connect(self) -> None:
        try:
            from hikvision_sdk import HCNetSDK
        except ImportError as e:
            raise RuntimeError(
                "hikvision-sdk not installed. Run: pip install hikvision-sdk"
            ) from e

        self._sdk = HCNetSDK(lib_path=self.hcnetsdk_path)
        self._device = self._cloud_login()

    def _cloud_login(self):
        """
        Attempt cloud login via serial + verification code.

        This is the part flagged in the module docstring as needing
        verification. `hikvision_sdk.HCNetSDK.login()` as published takes
        (ip, port, username, password) — a LAN-style login. If the version
        you install doesn't expose a serial/code cloud variant, you'll need
        to either:

          1. Check if a newer version of the package added cloud support
             (check PyPI changelog / GitHub issues), or
          2. Call NET_DVR_Login_V40 directly via ctypes using a LOGIN_INFO
             struct populated with your device serial and verification
             code (see Hikvision's official SDK demo code for the exact
             struct layout — it's in the C++ demos bundled with the SDK
             download).

        Raising NotImplementedError here on purpose rather than guessing
        at a wrong API and failing silently.
        """
        if not hasattr(self._sdk, "login_cloud"):
            raise NotImplementedError(
                "hikvision_sdk.HCNetSDK has no login_cloud() method in this "
                "version. See the CloudClient docstring in hik_client.py for "
                "how to proceed — you'll likely need a direct ctypes call "
                "into NET_DVR_Login_V40 with an ISUP/cloud LOGIN_INFO struct."
            )
        return self._sdk.login_cloud(
            serial=self.device_serial,
            code=self.verification_code,
        )

    def frames(self) -> Iterator[np.ndarray]:
        if self._device is None:
            self.connect()
        # Placeholder: once connect() succeeds, wire this up to whatever
        # frame-callback or pull API the SDK wrapper exposes for live
        # preview (commonly a callback-based API in HCNetSDK — you'll
        # likely need a small queue to bridge callback -> generator).
        raise NotImplementedError(
            "Frame pull not wired up yet — depends on the callback API "
            "shape once login_cloud() is confirmed working."
        )

    def close(self) -> None:
        if self._sdk is not None and self._device is not None:
            self._sdk.logout(self._device)


def build_client(mode: str, **kwargs):
    if mode == "rtsp":
        return RtspClient(rtsp_url=kwargs["rtsp_url"])
    elif mode == "cloud":
        return CloudClient(
            device_serial=kwargs["device_serial"],
            verification_code=kwargs["verification_code"],
            hcnetsdk_path=kwargs["hcnetsdk_path"],
        )
    raise ValueError(f"Unknown stream mode: {mode}")
