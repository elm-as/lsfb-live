"""Rendu visuel sobre et lisible pour l'affichage en direct OpenCV."""

from typing import Optional
import cv2
import numpy as np

from lsfb_live.capture.holistic_extractor import FrameLandmarks
from lsfb_live.recognition.live_engine import SignPrediction

# Connexions du torse supérieur MediaPipe Pose
POSE_CONNECTIONS = [
    (11, 12),  # Épaule gauche - Épaule droite
    (11, 13),  # Épaule gauche - Coude gauche
    (13, 15),  # Coude gauche - Poignet gauche
    (12, 14),  # Épaule droite - Coude droit
    (14, 16),  # Coude droit - Poignet droit
]

# Connexions des 5 doigts et de la paume de la main (21 repères)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pouce
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Majeur
    (0, 13), (13, 14), (14, 15), (15, 16), # Annulaire
    (0, 17), (17, 18), (18, 19), (19, 20), # Auriculaire
    (5, 9), (9, 13), (13, 17),             # Arc métacarpien de la paume
]
FINGERTIP_INDICES = {4, 8, 12, 16, 20}


def draw_skeleton_layer(
    frame: np.ndarray,
    landmarks: FrameLandmarks,
    is_mirrored: bool = True,
) -> None:
    """Trace les os et articulations du corps et des mains sur l'image d'affichage."""
    height, width = frame.shape[:2]

    def to_pixel(coords: np.ndarray, idx: int) -> tuple[int, int]:
        x_norm = (1.0 - coords[idx, 0]) if is_mirrored else coords[idx, 0]
        return int(x_norm * width), int(coords[idx, 1] * height)

    # 1. Tracé du torse supérieur
    if landmarks.has_pose:
        for p1, p2 in POSE_CONNECTIONS:
            pt1 = to_pixel(landmarks.pose, p1)
            pt2 = to_pixel(landmarks.pose, p2)
            cv2.line(frame, pt1, pt2, (190, 190, 190), 2, cv2.LINE_AA)
        for idx in [11, 12, 13, 14, 15, 16]:
            cv2.circle(frame, to_pixel(landmarks.pose, idx), 4, (0, 180, 255), -1, cv2.LINE_AA)

    # 2. Tracé haute précision des mains (gauche: cyan, droite: vert émeraude)
    hands_specs = [
        (landmarks.left_hand, (255, 200, 0)),
        (landmarks.right_hand, (0, 255, 140)),
    ]
    for hand_coords, color in hands_specs:
        if np.all(hand_coords == 0):
            continue
        for p1, p2 in HAND_CONNECTIONS:
            pt1 = to_pixel(hand_coords, p1)
            pt2 = to_pixel(hand_coords, p2)
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
        for idx in range(21):
            radius = 5 if idx in FINGERTIP_INDICES else 3
            pt_color = (60, 255, 255) if idx in FINGERTIP_INDICES else (255, 255, 255)
            cv2.circle(frame, to_pixel(hand_coords, idx), radius, pt_color, -1, cv2.LINE_AA)


def draw_hud_panel(
    frame: np.ndarray,
    prediction: Optional[SignPrediction],
    fps: float,
) -> None:
    """Affiche le bandeau d'information, le signe détecté et la confiance."""
    height, width = frame.shape[:2]

    # Bandeau supérieur semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (20, 20, 25), -1)
    cv2.rectangle(overlay, (0, height - 90), (280, height), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Indicateur FPS en haut à droite
    cv2.putText(
        frame,
        f"{fps:4.1f} FPS",
        (width - 110, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    if prediction is None:
        cv2.putText(frame, "Initialisation...", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2)
        return

    # Signe principal
    sign_color = (0, 240, 120) if prediction.is_active_sign else (120, 180, 240)
    display_text = f"{prediction.sign}  ({prediction.confidence * 100:4.1f}%)"
    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, sign_color, 2, cv2.LINE_AA)

    # Mini-liste des 3 meilleurs candidats en bas à gauche
    y_offset = height - 68
    cv2.putText(frame, "Candidats :", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    for sign_name, prob in prediction.top5[:3]:
        y_offset += 20
        cand_str = f"  {sign_name[:12]:<12} {prob * 100:4.1f}%"
        cv2.putText(frame, cand_str, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

