"""
DVR client using Hikvision's ISAPI (HTTP REST + XML) over HTTP Digest auth.

Two operations:

  search_recordings()   - POST /ISAPI/ContentMgmt/search
                           finds recorded segments for a track in a time
                           range, returns a list of RecordingMatch (each
                           with a playbackURI + start/end time).

  download_recording()  - POST /ISAPI/ContentMgmt/download
                           given a playbackURI, streams the segment to a
                           local file.

Track ID convention: channel N main stream = N*100 + 1, sub stream =
N*100 + 2 (e.g. channel 1 main = 101, channel 1 sub = 102).
"""
import logging
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import List

import requests
from requests.auth import HTTPDigestAuth

logger = logging.getLogger(__name__)

ISAPI_NS = "http://www.hikvision.com/ver20/XMLSchema"
ET.register_namespace("", ISAPI_NS)


class DvrError(Exception):
    """Raised when the DVR returns an error or an unparseable response."""


@dataclass
class RecordingMatch:
    playback_uri: str
    start_time: str
    end_time: str


def _iso(dt: str) -> str:
    """Accepts either an ISO string already, or normalizes via datetime."""
    return datetime.fromisoformat(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


class DvrClient:
    def __init__(self, host: str, username: str, password: str, port: int = 80, timeout: int = 30):
        self.base_url = f"http://{host}:{port}"
        self.auth = HTTPDigestAuth(username, password)
        self.timeout = timeout

    def search_recordings(
        self, track_id: int, start_time: str, end_time: str, max_results: int = 40
    ) -> List[RecordingMatch]:
        """
        Search for recordings on `track_id` between start_time and end_time
        (ISO 8601, e.g. "2026-07-28T00:00:00"). Returns matches in
        chronological order.
        """
        search_id = str(uuid.uuid4())
        body = f"""<?xml version="1.0" encoding="UTF-8"?>
<CMSearchDescription xmlns="{ISAPI_NS}">
  <searchID>{search_id}</searchID>
  <trackIDList>
    <trackID>{track_id}</trackID>
  </trackIDList>
  <timeSpanList>
    <timeSpan>
      <startTime>{_iso(start_time)}</startTime>
      <endTime>{_iso(end_time)}</endTime>
    </timeSpan>
  </timeSpanList>
  <maxResults>{max_results}</maxResults>
  <searchResultPostion>0</searchResultPostion>
  <metadataList>
    <metadataDescriptor>//recordType.meta.std-cgi.com</metadataDescriptor>
  </metadataList>
</CMSearchDescription>"""

        resp = requests.post(
            f"{self.base_url}/ISAPI/ContentMgmt/search",
            data=body.encode("utf-8"),
            auth=self.auth,
            headers={"Content-Type": "application/xml"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_search_response(resp.text)

    def _parse_search_response(self, xml_text: str) -> List[RecordingMatch]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise DvrError(f"Could not parse search response: {e}") from e

        ns = {"h": ISAPI_NS}

        # Some ISAPI errors come back as a generic <ResponseStatus> envelope
        # (statusCode/statusString) instead of a CMSearchResult with
        # responseStatus=false embedded in it. Catch both shapes rather than
        # silently treating an error as "zero results".
        local_tag = root.tag.split("}", 1)[-1]
        if local_tag == "ResponseStatus":
            code = root.findtext("h:statusCode", namespaces=ns) or "?"
            msg = root.findtext("h:statusString", namespaces=ns) or "unknown error"
            raise DvrError(f"DVR search failed (status {code}): {msg}")

        status = root.findtext("h:responseStatus", namespaces=ns)
        if status is not None and status.strip().lower() == "false":
            sub = root.findtext("h:responseStatusStrg", namespaces=ns) or "unknown"
            raise DvrError(f"DVR search failed: {sub}")

        matches: List[RecordingMatch] = []
        for item in root.findall(".//h:searchMatchItem", ns):
            uri = item.findtext("h:mediaSegmentDescriptor/h:playbackURI", namespaces=ns)
            start = item.findtext("h:timeSpan/h:startTime", namespaces=ns)
            end = item.findtext("h:timeSpan/h:endTime", namespaces=ns)
            if not uri:
                continue
            matches.append(RecordingMatch(playback_uri=uri, start_time=start or "", end_time=end or ""))

        matches.sort(key=lambda m: m.start_time)
        logger.info("Search returned %d recording(s)", len(matches))
        return matches

    def download_recording(self, match: RecordingMatch, dest_path: str) -> str:
        """Downloads a recording segment to dest_path, returns dest_path."""
        body = f"""<?xml version="1.0" encoding="UTF-8"?>
<downloadRequest xmlns="{ISAPI_NS}">
  <playbackURI>{match.playback_uri}</playbackURI>
</downloadRequest>"""

        resp = requests.post(
            f"{self.base_url}/ISAPI/ContentMgmt/download",
            data=body.encode("utf-8"),
            auth=self.auth,
            headers={"Content-Type": "application/xml"},
            timeout=self.timeout,
            stream=True,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "xml" in content_type.lower():
            # An XML response here means the DVR rejected the download and
            # sent an error body instead of video bytes.
            raise DvrError(f"Download rejected by DVR: {resp.text[:500]}")

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        logger.info("Downloaded recording to %s", dest_path)
        return dest_path
