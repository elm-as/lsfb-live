"""Extraction des repères MediaPipe Holistic pour chaque trame vidéo."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    HolisticLandmarkerResult,
    RunningMode,
)
import numpy as np


@dataclass(frozen=True)
class FrameLandmarks:
    """Repères extraits sur une unique trame vidéo.

    Attributes:
        pose: Coordonnées normalisées du corps (33, 3).
        left_hand: Coordonnées de la main gauche (21, 3).
        right_hand: Coordonnées de la main droite (21, 3).
        has_pose: Présence détectée d'un corps humain.
        has_hands: Au moins une main détectée avec succès.
    """

    pose: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    has_pose: bool
    has_hands: bool


def _extract_landmark_coords(landmark_list: list, expected_count: int) -> np.ndarray:
    """Convertit une liste de repères MediaPipe en tableau NumPy float32 (N, 3)."""
    if not landmark_list or len(landmark_list) < expected_count:
        return np.zeros((expected_count, 3), dtype=np.float32)

    coords = np.empty((expected_count, 3), dtype=np.float32)
    for idx in range(expected_count):
        lm = landmark_list[idx]
        coords[idx, 0] = lm.x
        coords[idx, 1] = lm.y
        coords[idx, 2] = lm.z
    return coords


class HolisticExtractor:
    """Enveloppe du modèle MediaPipe Holistic optimisé pour flux caméra temps réel."""

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Modèle MediaPipe introuvable : {model_path}")

        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.4,
        )
        self._landmarker = HolisticLandmarker.create_from_options(options)

    def extract_frame_landmarks(self, bgr_image: np.ndarray) -> FrameLandmarks:
        """Détecte le corps et les mains sur une image OpenCV BGR.

        Args:
            bgr_image: Image brute capturée par la webcam (H, W, 3) en BGR.

        Returns:
            Objet FrameLandmarks contenant les tableaux NumPy des repères.
        """
        rgb_image = bgr_image[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result: HolisticLandmarkerResult = self._landmarker.detect(mp_image)

        pose_coords = _extract_landmark_coords(result.pose_landmarks, 33)
        lh_coords = _extract_landmark_coords(result.left_hand_landmarks, 21)
        rh_coords = _extract_landmark_coords(result.right_hand_landmarks, 21)

        has_pose = bool(result.pose_landmarks)
        has_hands = bool(result.left_hand_landmarks or result.right_hand_landmarks)

        return FrameLandmarks(
            pose=pose_coords,
            left_hand=lh_coords,
            right_hand=rh_coords,
            has_pose=has_pose,
            has_hands=has_hands,
        )

    def close(self) -> None:
        """Libère les ressources natives C++ du landmarker."""
        self._landmarker.close()

