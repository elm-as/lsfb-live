"""Extraction haute précision des mains (21 pts/main) et du corps en mode VIDEO."""

from dataclasses import dataclass
from pathlib import Path
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
import numpy as np

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


@dataclass(frozen=True)
class FrameLandmarks:
    """Repères extraits sur une unique trame vidéo (espace caméra non inversé).

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


def _landmarks_to_numpy(landmark_list: list, expected_count: int) -> np.ndarray:
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


def _assign_left_right_hands(
    hand_landmarks_list: list,
    handedness_list: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Assigne les mains gauche et droite avec désambiguïsation géométrique."""
    lh = np.zeros((21, 3), dtype=np.float32)
    rh = np.zeros((21, 3), dtype=np.float32)
    if not hand_landmarks_list:
        return lh, rh

    parsed_hands: list[tuple[str, float, np.ndarray]] = []
    for idx, lm_list in enumerate(hand_landmarks_list):
        coords = _landmarks_to_numpy(lm_list, 21)
        label = "Right"
        score = 0.5
        if idx < len(handedness_list) and handedness_list[idx]:
            label = handedness_list[idx][0].category_name
            score = float(handedness_list[idx][0].score)
        parsed_hands.append((label, score, coords))

    if len(parsed_hands) == 1:
        label, _, coords = parsed_hands[0]
        if label == "Left":
            return coords, rh
        return lh, coords

    # Si 2 mains sont détectées avec le même label, on désambiguïse par l'axe X caméra :
    # En vue caméra non inversée, la main gauche du signeur est à droite de l'image (X plus grand).
    h0_label, h0_score, h0_coords = parsed_hands[0]
    h1_label, h1_score, h1_coords = parsed_hands[1]
    if h0_label != h1_label:
        if h0_label == "Left":
            return h0_coords, h1_coords
        return h1_coords, h0_coords

    if h0_coords[0, 0] >= h1_coords[0, 0]:
        return h0_coords, h1_coords
    return h1_coords, h0_coords


def _anchor_pose_wrists_to_hands(
    pose: np.ndarray,
    lh: np.ndarray,
    rh: np.ndarray,
) -> np.ndarray:
    """Ancre les poignets et coudes du corps sur les vraies mains détectées."""
    anchored = pose.copy()
    has_lh = not np.all(lh == 0)
    has_rh = not np.all(rh == 0)

    if has_lh:
        anchored[15, :2] = lh[0, :2]
        if anchored[13, 1] > 0.90 or anchored[13, 1] < 0.05:
            anchored[13, :2] = 0.5 * (anchored[11, :2] + lh[0, :2])
            anchored[13, 1] += 0.08

    if has_rh:
        anchored[16, :2] = rh[0, :2]
        if anchored[14, 1] > 0.90 or anchored[14, 1] < 0.05:
            anchored[14, :2] = 0.5 * (anchored[12, :2] + rh[0, :2])
            anchored[14, 1] += 0.08

    return anchored


class HolisticExtractor:
    """Extracteur combinant HandLandmarker dédié (BlazePalm VIDEO) et PoseLandmarker."""

    def __init__(self, model_path: Path):
        models_dir = model_path.parent if model_path.suffix else model_path
        hand_model_path = models_dir / "hand_landmarker.task"
        pose_model_path = models_dir / "pose_landmarker_lite.task"
        ensure_holistic_model_present(model_path)

        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.30,
            min_hand_presence_confidence=0.30,
            min_tracking_confidence=0.30,
        )
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.40,
            min_tracking_confidence=0.40,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(hand_options)
        self._pose_landmarker = PoseLandmarker.create_from_options(pose_options)
        self._last_timestamp_ms = 0
        self._prev_lh = np.zeros((21, 3), dtype=np.float32)
        self._prev_rh = np.zeros((21, 3), dtype=np.float32)
        self._lh_missing_count = 10
        self._rh_missing_count = 10

        default_pose = np.zeros((33, 3), dtype=np.float32)
        default_pose[0] = [0.50, 0.32, 0.0]   # Nez canonique
        default_pose[11] = [0.62, 0.68, 0.0]  # Épaule gauche (vue caméra)
        default_pose[12] = [0.38, 0.68, 0.0]  # Épaule droite (vue caméra)
        default_pose[13] = [0.66, 0.90, 0.0]  # Coude gauche
        default_pose[14] = [0.34, 0.90, 0.0]  # Coude droit
        default_pose[15] = [0.62, 0.95, 0.0]  # Poignet gauche
        default_pose[16] = [0.38, 0.95, 0.0]  # Poignet droit
        self._prev_pose = default_pose

    def extract_frame_landmarks(self, bgr_image: np.ndarray) -> FrameLandmarks:
        """Détecte les 21 phalanges par main et le buste sur une image BGR non inversée."""
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        now_ms = int(time.perf_counter() * 1000)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms

        hand_result = self._hand_landmarker.detect_for_video(mp_image, now_ms)
        pose_result = self._pose_landmarker.detect_for_video(mp_image, now_ms)

        has_pose = bool(pose_result.pose_landmarks)
        if has_pose:
            pose_coords = _landmarks_to_numpy(pose_result.pose_landmarks[0], 33)
            self._prev_pose = pose_coords.copy()
        else:
            pose_coords = self._prev_pose.copy()

        lh_coords, rh_coords = _assign_left_right_hands(
            hand_result.hand_landmarks,
            hand_result.handedness,
        )

        # Maintien court-terme (2 trames max) contre le flou cinétique rapide des doigts
        if not np.all(lh_coords == 0):
            self._prev_lh = lh_coords
            self._lh_missing_count = 0
        elif self._lh_missing_count < 2:
            self._lh_missing_count += 1
            lh_coords = self._prev_lh

        if not np.all(rh_coords == 0):
            self._prev_rh = rh_coords
            self._rh_missing_count = 0
        elif self._rh_missing_count < 2:
            self._rh_missing_count += 1
            rh_coords = self._prev_rh

        pose_coords = _anchor_pose_wrists_to_hands(pose_coords, lh_coords, rh_coords)

        has_hands = not np.all(lh_coords == 0) or not np.all(rh_coords == 0)
        return FrameLandmarks(
            pose=pose_coords,
            left_hand=lh_coords,
            right_hand=rh_coords,
            has_pose=has_pose or has_hands,
            has_hands=has_hands,
        )

    def close(self) -> None:
        """Libère les ressources natives C++ des détecteurs."""
        self._hand_landmarker.close()
        self._pose_landmarker.close()


def ensure_holistic_model_present(model_path: Path) -> None:
    """Vérifie et télécharge les modèles HandLandmarker et PoseLandmarker."""
    models_dir = model_path.parent if model_path.suffix else model_path
    models_dir.mkdir(parents=True, exist_ok=True)

    hand_path = models_dir / "hand_landmarker.task"
    if not hand_path.is_file():
        print(f"Téléchargement du détecteur de mains haute précision ({HAND_MODEL_URL})...")
        urllib.request.urlretrieve(HAND_MODEL_URL, hand_path)

    pose_path = models_dir / "pose_landmarker_lite.task"
    if not pose_path.is_file():
        print(f"Téléchargement du détecteur de pose rapide ({POSE_MODEL_URL})...")
        urllib.request.urlretrieve(POSE_MODEL_URL, pose_path)
