# hik-face-detect

[![GitHub repo](https://img.shields.io/badge/GitHub-vchhajed%2Fhik--face--detect-blue?logo=github)](https://github.com/vchhajed/hik-face-detect)

Run edge face detection against a Hikvision IP camera or DVR — either a live
stream, or recorded DVR footage — and fire a notification (Telegram and/or a
dashboard) when a face is detected.

## How it works

There are **two ways** this repo can reach your camera, and they matter
differently depending on where this code runs:

| Mode | Works when... | Setup effort |
|---|---|---|
| `rtsp` | The machine running this code is on the same network as the camera, or reaches it via VPN/Tailscale | Low — just needs OpenCV |
| `cloud` | The machine is anywhere on the internet, using Hik-Connect P2P + the camera's serial number and verification code | Higher — needs Hikvision's official HCNetSDK binaries, and the cloud login parameters may need adjusting against your SDK version (see `src/hik_client.py` for notes) |

**Honest note on `cloud` mode:** Hikvision's cloud P2P protocol is
proprietary and only partially documented, even in their official SDK. The
`hikvision-sdk` Python wrapper this repo uses is built primarily for LAN
login. Cloud login (via serial + verification code, no IP) may require a
slightly different SDK call than what's wired up here — I've left this
clearly marked in `hik_client.py` so it's easy to patch once you've got the
SDK downloaded and can see what's actually exposed. Budget some debugging
time for this part specifically.

If you don't want to deal with that uncertainty, run this on a device on
your home network (or reachable via VPN) and use `rtsp` mode — it's the
"just works" path.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/vchhajed/hik-face-detect.git
cd hik-face-detect
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the face detection model

```bash
mkdir -p models
curl -L -o models/face_detection_yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```
# --- Camera connection ---
STREAM_MODE=rtsp                  # rtsp | cloud

# rtsp mode
RTSP_URL=rtsp://admin:yourpassword@192.168.1.50:554/Streaming/Channels/102

# cloud mode (only needed if STREAM_MODE=cloud)
DEVICE_SERIAL=DS-2CD1041G2-LIU20250310AAWRFX4320713
VERIFICATION_CODE=Vaibhav123
HCNETSDK_PATH=/path/to/libhcnetsdk.so   # from the Hikvision SDK download

# --- Detection ---
DETECTION_CONFIDENCE=0.8
NOTIFY_COOLDOWN_SECONDS=30         # don't spam notifications

# --- Notifications (Telegram) ---
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

To get a Telegram bot token: message `@BotFather` on Telegram, `/newbot`,
follow the prompts. To get your chat ID: message your new bot anything, then
visit `https://api.telegram.org/bot<TOKEN>/getUpdates` and read `chat.id`
from the response.

### 4. If using `cloud` mode: get HCNetSDK

- Register a free account at `open.hikvision.com`
- Download the Network SDK (HCNetSDK) for your OS
- Point `HCNETSDK_PATH` in `.env` at the shared library (`.so` on Linux,
  `.dll` on Windows)

### 5. Run

```bash
python -m src.main
```

You should see frame processing logs, and a Telegram message whenever a face
is detected (rate-limited by `NOTIFY_COOLDOWN_SECONDS`).

## DVR mode (recorded footage instead of live stream)

Instead of (or in addition to) the live RTSP loop, you can pull recorded
footage straight from the DVR over its ISAPI HTTP interface (auth via HTTP
Digest — no VPN/live-stream latency to worry about, but the machine running
this still needs to reach the DVR's IP, whether that's LAN, VPN, or a
port-forwarded static IP).

```bash
python -m src.dvr_batch \
  --host 192.168.1.5 --username admin --password 12345abc \
  --track 101 \
  --start "2026-07-28T00:00:00" --end "2026-07-28T23:59:59" \
  --dry-run   # remove this flag once the search results look right
```

`--track` follows Hikvision's convention: channel N main stream = `N*100 +
1` (channel 1 = `101`), sub-stream = `N*100 + 2`. Check the DVR's web UI for
its actual channel count/numbering before assuming these.

What it does, in order: `search_recordings()` (ISAPI `ContentMgmt/search`)
finds matching segments in the time range; each match is downloaded
(`ContentMgmt/download`) to a temp file; `FaceDetector` runs over sampled
frames (`--sample-every-n-frames`, default 5); any frame with a detection is
sent through the same notifier stack as the live path; each downloaded
segment is deleted once processed. Detection/notifier settings (confidence
threshold, cooldown, Telegram/dashboard config) still come from `.env` — only
the DVR connection itself is passed via CLI flags.

If a search with real footage in range returns zero matches, the most likely
cause is `--track` not matching this DVR's actual channel numbering — check
the web UI's channel list first before assuming there's no footage.

## Project structure

```
src/
  main.py           # live stream loop: capture -> detect -> notify
  hik_client.py      # camera connection (rtsp or cloud)
  face_detector.py    # YuNet face detection wrapper
  notifier.py         # Telegram / dashboard notification senders
  config.py           # loads .env
  dvr_client.py        # DVR ISAPI: search_recordings(), download_recording()
  process_file.py       # runs FaceDetector over a local video file
  dvr_batch.py           # CLI: search DVR range -> download -> detect -> notify -> cleanup
models/
  (face_detection_yunet.onnx goes here — not committed, see setup step 2)
.env.example
requirements.txt
```

## Extending

- Swap `notifier.py` for email/Slack/webhook — same interface, just replace
  `send()`.
- Swap `face_detector.py` for a heavier model (RetinaFace, YOLOv8-face) if
  you move this to GPU hardware — `detect(frame) -> list[BoundingBox]` is the
  only contract `main.py` depends on.
- Add frame saving on detection by extending the callback in `main.py`.
