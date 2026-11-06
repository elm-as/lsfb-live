"""Gestion du flux vidéo webcam et calcul du débit d'images (FPS)."""

import time
import cv2
import numpy as np


class WebcamStream:
    """Capture vidéo OpenCV encapsulée avec gestion d'erreurs et mesure de fréquence.

    Args:
        camera_index: Index matériel de la webcam (0 par défaut).
        width: Largeur souhaitée de la capture (640 par défaut).
        height: Hauteur souhaitée de la capture (480 par défaut).
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Repli sur le backend générique si DirectShow échoue
            self._cap = cv2.VideoCapture(camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la webcam à l'index {camera_index}.")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._prev_timestamp = time.perf_counter()
        self._current_fps = 0.0

    def read_frame(self) -> tuple[bool, np.ndarray]:
        """Lit la prochaine trame vidéo et actualise la cadence mesurée."""
        success, frame = self._cap.read()
        if not success:
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        now = time.perf_counter()
        delta = now - self._prev_timestamp
        self._prev_timestamp = now
        if delta > 0:
            self._current_fps = 0.9 * self._current_fps + 0.1 * (1.0 / delta)

        return True, frame

    @property
    def fps(self) -> float:
        """Cadence instantanée mesurée en trames par seconde."""
        return self._current_fps

    def release(self) -> None:
        """Libère le périphérique matériel de capture."""
        if self._cap.isOpened():
            self._cap.release()

