"""Lightweight CPU face detection using OpenCV's YuNet model."""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float


class FaceDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self._detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold=confidence_threshold
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)

        if faces is None:
            return []

        results = []
        for face in faces:
            x, y, fw, fh = face[:4].astype(int)
            confidence = float(face[-1])
            if confidence >= self.confidence_threshold:
                results.append(Detection(x=x, y=y, w=fw, h=fh, confidence=confidence))
        return results
