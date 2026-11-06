"""Rendu visuel pour le mode entraînement interactif (Coach LSFB)."""

from typing import Optional
import cv2
import numpy as np

from lsfb_live.capture.holistic_extractor import FrameLandmarks
from lsfb_live.display.visualizer import HAND_CONNECTIONS, POSE_CONNECTIONS


def draw_centered_skeleton(
    canvas: np.ndarray,
    landmarks: FrameLandmarks,
) -> None:
    """Dessine le squelette normalisé et centré sur le canvas de référence.

    Args:
        canvas: Image destination BGR (H, W, 3).
        landmarks: Repères normalisés de la trame.
    """
    height, width = canvas.shape[:2]
    if not landmarks.has_pose:
        return

    pose = landmarks.pose
    mid_shoulder_x = float((pose[11, 0] + pose[12, 0]) / 2.0)
    mid_shoulder_y = float((pose[11, 1] + pose[12, 1]) / 2.0)
    shoulder_dist = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
    if shoulder_dist < 1e-4:
        return

    scale = (width * 0.28) / shoulder_dist
    target_center_x = width * 0.50
    target_center_y = height * 0.50

    def to_canvas_pixel(point: np.ndarray) -> tuple[int, int]:
        # Inversion miroir de l'axe X pour que le coach fonctionne comme un miroir
        px = int(target_center_x - (point[0] - mid_shoulder_x) * scale)
        py = int(target_center_y + (point[1] - mid_shoulder_y) * scale)
        return max(0, min(width - 1, px)), max(0, min(height - 1, py))

    # 1. Tracé de la tête et du cou
    nose_px = to_canvas_pixel(pose[0])
    neck_px = to_canvas_pixel(np.array([mid_shoulder_x, mid_shoulder_y, 0.0]))
    head_radius = max(18, int(shoulder_dist * scale * 0.30))

    cv2.line(canvas, neck_px, nose_px, (140, 150, 170), 3, cv2.LINE_AA)
    cv2.circle(canvas, nose_px, head_radius, (35, 40, 50), -1, cv2.LINE_AA)
    cv2.circle(canvas, nose_px, head_radius, (160, 180, 210), 2, cv2.LINE_AA)
    # Yeux
    eye_l = to_canvas_pixel(pose[2])
    eye_r = to_canvas_pixel(pose[5])
    cv2.circle(canvas, eye_l, 3, (220, 230, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, eye_r, 3, (220, 230, 255), -1, cv2.LINE_AA)

    # 2. Buste et bras (ambre chaud)
    for p1, p2 in POSE_CONNECTIONS:
        pt1 = to_canvas_pixel(pose[p1])
        pt2 = to_canvas_pixel(pose[p2])
        cv2.line(canvas, pt1, pt2, (80, 180, 245), 3, cv2.LINE_AA)
    for idx in [11, 12, 13, 14, 15, 16]:
        cv2.circle(canvas, to_canvas_pixel(pose[idx]), 5, (255, 220, 100), -1, cv2.LINE_AA)

    # 3. Mains connectées aux poignets
    hands_specs = [
        (landmarks.left_hand, pose[15], (255, 210, 80)),
        (landmarks.right_hand, pose[16], (80, 240, 160)),
    ]
    for hand_coords, wrist_coords, color in hands_specs:
        if np.all(hand_coords == 0) or np.isnan(hand_coords).any():
            continue
        # Raccordement poignet-main
        cv2.line(canvas, to_canvas_pixel(wrist_coords), to_canvas_pixel(hand_coords[0]), color, 2, cv2.LINE_AA)
        for p1, p2 in HAND_CONNECTIONS:
            pt1 = to_canvas_pixel(hand_coords[p1])
            pt2 = to_canvas_pixel(hand_coords[p2])
            cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
        for idx in range(21):
            cv2.circle(canvas, to_canvas_pixel(hand_coords[idx]), 3, (255, 255, 255), -1, cv2.LINE_AA)


def build_coach_reference_card(
    canvas: np.ndarray,
    sign_name: str,
    description: str,
    progress_ratio: float,
) -> None:
    """Trace les bandeaux informatifs, la description gestuelle et la jauge de progression.

    Args:
        canvas: Image canvas BGR (H, W, 3).
        sign_name: Nom du signe affiché.
        description: Conseil gestuel en français.
        progress_ratio: Avancement temporel de la séquence (0.0 à 1.0).
    """
    height, width = canvas.shape[:2]

    # Bandeau supérieur semi-transparent
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (32, 34, 40), -1)
    cv2.rectangle(overlay, (0, height - 70), (width, height), (32, 34, 40), -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)

    # Titre et badge
    cv2.putText(canvas, "MODE ENTRAINEMENT (COACH LSFB)", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 180, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, sign_name, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # Conseil gestuel
    cv2.putText(canvas, description, (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 200), 1, cv2.LINE_AA)

    # Jauge temporelle de l'animation
    bar_width = int(width * np.clip(progress_ratio, 0.0, 1.0))
    cv2.line(canvas, (0, 99), (bar_width, 99), (0, 220, 160), 3, cv2.LINE_AA)

    # Raccourcis clavier en bas
    cv2.putText(canvas, "[N] Signe suivant  |  [P] Precedent  |  [Q] Quitter", (16, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)


def assemble_practice_split_screen(
    ref_canvas: np.ndarray,
    webcam_frame: np.ndarray,
    is_target_recognized: bool,
    target_confidence: float,
) -> np.ndarray:
    """Assemble côte à côte le canvas de référence et le flux webcam avec rétroaction visuelle.

    Args:
        ref_canvas: Panneau de gauche avec l'animation de référence (H, W, 3).
        webcam_frame: Panneau de droite avec la webcam utilisateur (H, W, 3).
        is_target_recognized: Vrai si le modèle détecte le signe cible avec confiance suffisante.
        target_confidence: Probabilité estimée pour le signe cible.

    Returns:
        Image BGR combinée de forme (H, W*2, 3).
    """
    height = ref_canvas.shape[0]
    webcam_resized = cv2.resize(webcam_frame, (ref_canvas.shape[1], height))

    # Liseré de séparation central
    cv2.line(ref_canvas, (ref_canvas.shape[1] - 1, 0), (ref_canvas.shape[1] - 1, height), (60, 65, 75), 2)

    # Rétroaction de validation sur le côté webcam
    if is_target_recognized:
        cv2.rectangle(webcam_resized, (0, 0), (webcam_resized.shape[1], height), (0, 220, 100), 6)
        cv2.rectangle(webcam_resized, (20, 20), (webcam_resized.shape[1] - 20, 80), (10, 80, 30), -1)
        feedback_text = f"BRAVO ! SIGNE VALIDE ({target_confidence * 100:.1f}%)"
        cv2.putText(webcam_resized, feedback_text, (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    return np.hstack([ref_canvas, webcam_resized])

