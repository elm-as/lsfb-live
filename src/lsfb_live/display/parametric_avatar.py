"""Avatar vectoriel cinématique et paramétrique (Base v3 Text-to-Sign).

Génère un rendu anatomique propre, fluide et proportionné d'un avatar humain,
piloté par des équations cinématiques mathématiques pour chaque signe canonique.
"""

from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


@dataclass(frozen=True)
class AvatarPoseState:
    """État cinématique instantané de l'avatar pour une trame donnée.

    Attributes:
        left_wrist: Position (x, y) du poignet gauche normalisée (0 à 1).
        right_wrist: Position (x, y) du poignet droit normalisée (0 à 1).
        left_hand_shape: Forme de la main ('open', 'index', 'fist', 'roof', 'rest').
        right_hand_shape: Forme de la main ('open', 'index', 'fist', 'roof', 'rest').
    """

    left_wrist: tuple[float, float]
    right_wrist: tuple[float, float]
    left_hand_shape: str = "open"
    right_hand_shape: str = "open"


def compute_canonical_sign_pose(sign_name: str, progress: float) -> AvatarPoseState:
    """Calcule la cinématique paramétrique mathématique d'un signe à l'instant t.

    Args:
        sign_name: Glose du signe ('RENCONTRER', 'DIFFERENT', 'MAISON', etc.).
        progress: Avancement dans le cycle du geste (0.0 à 1.0).

    Returns:
        AvatarPoseState contenant les coordonnées des poignets et les configurations.
    """
    # Phase d'oscillation fluide : aller-retour avec amorti sinusoïdal
    t = float(np.clip(progress, 0.0, 1.0))
    smooth_t = float(0.5 - 0.5 * np.cos(np.pi * t))  # Easing in-out

    if sign_name == "RENCONTRER":
        # Deux index verticaux qui partent des côtés et se rejoignent au centre de la poitrine
        # Position de départ : écartés (x=0.35 et x=0.65), y=0.55
        # Position d'arrivée : joints au centre (x=0.48 et x=0.52), y=0.52
        lx = 0.35 + (0.48 - 0.35) * smooth_t
        rx = 0.65 + (0.52 - 0.65) * smooth_t
        ly = 0.56 - 0.04 * np.sin(np.pi * t)
        ry = 0.56 - 0.04 * np.sin(np.pi * t)
        return AvatarPoseState((lx, ly), (rx, ry), "index", "index")

    if sign_name == "DIFFERENT":
        # Deux index qui démarrent joints au centre et s'écartent vivement
        lx = 0.48 - 0.18 * smooth_t
        rx = 0.52 + 0.18 * smooth_t
        ly = 0.52 + 0.02 * np.sin(np.pi * t)
        ry = 0.52 + 0.02 * np.sin(np.pi * t)
        return AvatarPoseState((lx, ly), (rx, ry), "index", "index")

    if sign_name == "MAISON":
        # Deux mains obliques qui montent et joignent le bout des doigts en toit
        lx = 0.38 + 0.08 * smooth_t
        rx = 0.62 - 0.08 * smooth_t
        ly = 0.62 - 0.16 * smooth_t
        ry = 0.62 - 0.16 * smooth_t
        return AvatarPoseState((lx, ly), (rx, ry), "roof", "roof")

    if sign_name == "CORPS":
        # Deux mains ouvertes descendant le long du buste, du haut vers le bas
        lx = 0.40
        rx = 0.60
        ly = 0.45 + 0.22 * smooth_t
        ry = 0.45 + 0.22 * smooth_t
        return AvatarPoseState((lx, ly), (rx, ry), "open", "open")

    if sign_name == "AVANCER":
        # Deux mains ouvertes poussant vers l'avant (vers la caméra)
        scale_offset = 0.06 * smooth_t
        lx = 0.42 - scale_offset
        rx = 0.58 + scale_offset
        ly = 0.54 - 0.05 * smooth_t
        ry = 0.54 - 0.05 * smooth_t
        return AvatarPoseState((lx, ly), (rx, ry), "open", "open")

    if sign_name == "MONTRER":
        # Main droite index pointé vers l'avant, main gauche au repos
        rx = 0.50 + 0.08 * smooth_t
        ry = 0.52 - 0.06 * smooth_t
        return AvatarPoseState((0.35, 0.72), (rx, ry), "rest", "index")

    if sign_name == "INTERPRETE":
        # Mains alternées tournant l'une autour de l'autre
        angle = 2.0 * np.pi * t
        lx = 0.44 + 0.06 * np.cos(angle)
        ly = 0.52 + 0.06 * np.sin(angle)
        rx = 0.56 + 0.06 * np.cos(angle + np.pi)
        ry = 0.52 + 0.06 * np.sin(angle + np.pi)
        return AvatarPoseState((lx, ly), (rx, ry), "open", "open")

    # Position de repos par défaut
    return AvatarPoseState((0.36, 0.70), (0.64, 0.70), "open", "open")


def draw_styled_hand(
    canvas: np.ndarray,
    wrist_px: tuple[int, int],
    hand_shape: str,
    is_left: bool,
) -> None:
    """Dessine une main anatomique stylisée selon la configuration gestuelle."""
    wx, wy = wrist_px
    direction = -1 if is_left else 1
    hand_color = (255, 220, 90) if is_left else (90, 245, 175)

    if hand_shape == "index":
        # Paume compacte + index dressé pointé vers le haut
        cv2.circle(canvas, (wx, wy), 9, (45, 55, 70), -1, cv2.LINE_AA)
        cv2.circle(canvas, (wx, wy), 9, hand_color, 2, cv2.LINE_AA)
        # Index tendu
        finger_tip = (wx, wy - 32)
        cv2.line(canvas, (wx, wy - 6), finger_tip, hand_color, 4, cv2.LINE_AA)
        cv2.circle(canvas, finger_tip, 4, (255, 255, 255), -1, cv2.LINE_AA)
        # Pouce replié
        cv2.line(canvas, (wx, wy), (wx + direction * 10, wy - 8), hand_color, 3, cv2.LINE_AA)
        return

    if hand_shape == "roof":
        # Main oblique à 45°
        tip_x = wx + (16 if is_left else -16)
        tip_y = wy - 24
        cv2.line(canvas, (wx, wy), (tip_x, tip_y), hand_color, 4, cv2.LINE_AA)
        cv2.circle(canvas, (tip_x, tip_y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        return

    if hand_shape == "rest":
        cv2.circle(canvas, (wx, wy), 7, (80, 90, 105), -1, cv2.LINE_AA)
        return

    # Main ouverte standard (open) : paume + 5 doigts déployés
    cv2.circle(canvas, (wx, wy), 10, (40, 50, 65), -1, cv2.LINE_AA)
    cv2.circle(canvas, (wx, wy), 10, hand_color, 2, cv2.LINE_AA)
    finger_angles = [-35, -15, 0, 15, 35] if not is_left else [35, 15, 0, -15, -35]
    for angle_deg in finger_angles:
        rad = np.radians(angle_deg)
        fx = int(wx + 22 * np.sin(rad))
        fy = int(wy - 22 * np.cos(rad))
        cv2.line(canvas, (wx, wy), (fx, fy), hand_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (fx, fy), 3, (255, 255, 255), -1, cv2.LINE_AA)


def render_parametric_avatar_frame(
    canvas: np.ndarray,
    sign_name: str,
    progress: float,
) -> None:
    """Génère le rendu complet de l'avatar paramétrique avec proportions anatomiques réelles.

    Args:
        canvas: Image destination BGR (H, W, 3).
        sign_name: Glose du signe.
        progress: Avancement dans l'animation (0.0 à 1.0).
    """
    height, width = canvas.shape[:2]
    pose = compute_canonical_sign_pose(sign_name, progress)

    # 1. Proportions anatomiques du buste
    center_x = width // 2
    head_center = (center_x, int(height * 0.22))
    head_radius = 42

    neck_top = (center_x, int(height * 0.22 + head_radius))
    neck_base = (center_x, int(height * 0.35))

    shoulder_l = (center_x - 100, int(height * 0.36))
    shoulder_r = (center_x + 100, int(height * 0.36))
    waist_l = (center_x - 70, int(height * 0.72))
    waist_r = (center_x + 70, int(height * 0.72))

    # 2. Dessin du torse stylisé (silhouette sobre gris foncé)
    torso_polygon = np.array([shoulder_l, shoulder_r, waist_r, waist_l], dtype=np.int32)
    cv2.fillPoly(canvas, [torso_polygon], (30, 34, 42), lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [torso_polygon], True, (65, 75, 90), 2, lineType=cv2.LINE_AA)

    # 3. Dessin du cou et de la tête
    cv2.line(canvas, neck_top, neck_base, (160, 175, 195), 4, cv2.LINE_AA)
    cv2.circle(canvas, head_center, head_radius, (38, 44, 55), -1, cv2.LINE_AA)
    cv2.circle(canvas, head_center, head_radius, (170, 190, 215), 2, cv2.LINE_AA)

    # Yeux expressifs
    eye_offset_x = 16
    eye_offset_y = 6
    cv2.circle(canvas, (head_center[0] - eye_offset_x, head_center[1] - eye_offset_y), 4, (240, 245, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (head_center[0] + eye_offset_x, head_center[1] - eye_offset_y), 4, (240, 245, 255), -1, cv2.LINE_AA)

    # Clavicules
    cv2.line(canvas, shoulder_l, neck_base, (85, 100, 120), 2, cv2.LINE_AA)
    cv2.line(canvas, shoulder_r, neck_base, (85, 100, 120), 2, cv2.LINE_AA)

    # 4. Bras gauche et droit avec coudes cinématiques
    lw_px = (int(pose.left_wrist[0] * width), int(pose.left_wrist[1] * height))
    rw_px = (int(pose.right_wrist[0] * width), int(pose.right_wrist[1] * height))

    def compute_elbow(shoulder: tuple[int, int], wrist: tuple[int, int], outwards: int) -> tuple[int, int]:
        mid_x = (shoulder[0] + wrist[0]) // 2
        mid_y = (shoulder[1] + wrist[1]) // 2
        return (mid_x + outwards * 28, max(shoulder[1] + 25, mid_y + 15))

    elbow_l = compute_elbow(shoulder_l, lw_px, outwards=-1)
    elbow_r = compute_elbow(shoulder_r, rw_px, outwards=1)

    # Bras gauche (ambre)
    cv2.line(canvas, shoulder_l, elbow_l, (80, 180, 245), 4, cv2.LINE_AA)
    cv2.line(canvas, elbow_l, lw_px, (80, 180, 245), 4, cv2.LINE_AA)
    cv2.circle(canvas, elbow_l, 6, (255, 220, 100), -1, cv2.LINE_AA)

    # Bras droit (ambre)
    cv2.line(canvas, shoulder_r, elbow_r, (80, 180, 245), 4, cv2.LINE_AA)
    cv2.line(canvas, elbow_r, rw_px, (80, 180, 245), 4, cv2.LINE_AA)
    cv2.circle(canvas, elbow_r, 6, (255, 220, 100), -1, cv2.LINE_AA)

    # 5. Mains articulées stylisées
    draw_styled_hand(canvas, lw_px, pose.left_hand_shape, is_left=True)
    draw_styled_hand(canvas, rw_px, pose.right_hand_shape, is_left=False)

