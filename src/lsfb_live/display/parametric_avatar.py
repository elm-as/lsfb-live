"""Avatar vectoriel cinématique et paramétrique (Base v3 Text-to-Sign).

Génère un rendu anatomique propre, fluide et proportionné d'un avatar humain,
piloté par des équations cinématiques mathématiques pour chaque signe canonique
ainsi qu'un interpolateur continu de séquences de signes (Text-to-Sign).
"""

from dataclasses import dataclass
import cv2
import numpy as np

from lsfb_live.avatar.text_to_sign_sequencer import get_sign_rig_at_progress
from lsfb_live.display.rig_renderer_cv import render_rig_mannequin_cv


@dataclass(frozen=True)
class AvatarPoseState:
    """État cinématique instantané de l'avatar pour une trame donnée.

    Attributes:
        left_wrist: Position (x, y) du poignet gauche normalisée (0 à 1).
        right_wrist: Position (x, y) du poignet droit normalisée (0 à 1).
        left_hand_shape: Forme de la main ('open', 'index', 'l_shape', 'fist', 'roof', 'rest').
        right_hand_shape: Forme de la main ('open', 'index', 'l_shape', 'fist', 'roof', 'rest').
    """

    left_wrist: tuple[float, float]
    right_wrist: tuple[float, float]
    left_hand_shape: str = "open"
    right_hand_shape: str = "open"


def compute_canonical_sign_pose(sign_name: str, progress: float) -> AvatarPoseState:
    """Calcule la cinématique paramétrique mathématique d'un signe à l'instant t.

    Args:
        sign_name: Glose du signe ('AIMER', 'PAPA', 'OUBLIER', 'RENCONTRER', etc.).
        progress: Avancement dans le cycle du geste (0.0 à 1.0).

    Returns:
        AvatarPoseState contenant les coordonnées des poignets et les configurations.
    """
    t = float(np.clip(progress, 0.0, 1.0))
    smooth_t = float(0.5 - 0.5 * np.cos(np.pi * t))
    pulse = float(np.sin(2.0 * np.pi * t) ** 2)

    if sign_name == "AIMER":
        # Main dominante ouverte posée sur le cœur/torse glissant légèrement avec pression
        rx = 0.50 - 0.03 * smooth_t
        ry = 0.48 + 0.05 * smooth_t
        return AvatarPoseState((0.34, 0.72), (rx, ry), "rest", "open")

    if sign_name == "PAPA":
        # Main en configuration 'L' (pouce + index tendus) tapotant près du menton/joue
        rx = 0.54 - 0.03 * pulse
        ry = 0.31 + 0.02 * pulse
        return AvatarPoseState((0.34, 0.72), (rx, ry), "rest", "l_shape")

    if sign_name == "OUBLIER":
        # Main passant devant la tempe/front et se refermant en poing ('fist')
        rx = 0.52 + 0.10 * smooth_t
        ry = 0.25 + 0.03 * smooth_t
        shape = "open" if t < 0.35 else "fist"
        return AvatarPoseState((0.34, 0.72), (rx, ry), "rest", shape)

    if sign_name == "COMPRENDRE":
        # Main fermée près de la tempe qui s'ouvre vers le haut en 'L' / index
        rx = 0.57 + 0.02 * smooth_t
        ry = 0.27 - 0.03 * smooth_t
        shape = "fist" if t < 0.30 else "l_shape"
        return AvatarPoseState((0.34, 0.72), (rx, ry), "rest", shape)

    if sign_name == "RENCONTRER":
        lx = 0.35 + (0.48 - 0.35) * smooth_t
        rx = 0.65 + (0.52 - 0.65) * smooth_t
        ly = 0.56 - 0.04 * np.sin(np.pi * t)
        ry = 0.56 - 0.04 * np.sin(np.pi * t)
        return AvatarPoseState((lx, ly), (rx, ry), "index", "index")

    if sign_name == "DIFFERENT":
        lx = 0.48 - 0.18 * smooth_t
        rx = 0.52 + 0.18 * smooth_t
        ly = 0.52 + 0.02 * np.sin(np.pi * t)
        ry = 0.52 + 0.02 * np.sin(np.pi * t)
        return AvatarPoseState((lx, ly), (rx, ry), "index", "index")

    if sign_name == "MAISON":
        lx = 0.38 + 0.08 * smooth_t
        rx = 0.62 - 0.08 * smooth_t
        ly = 0.62 - 0.16 * smooth_t
        ry = 0.62 - 0.16 * smooth_t
        return AvatarPoseState((lx, ly), (rx, ry), "roof", "roof")

    if sign_name == "CORPS":
        ly = 0.45 + 0.22 * smooth_t
        ry = 0.45 + 0.22 * smooth_t
        return AvatarPoseState((0.40, ly), (0.60, ry), "open", "open")

    if sign_name == "AVANCER":
        scale_offset = 0.06 * smooth_t
        lx = 0.42 - scale_offset
        rx = 0.58 + scale_offset
        ly = 0.54 - 0.05 * smooth_t
        return AvatarPoseState((lx, ly), (rx, ly), "open", "open")

    if sign_name == "MONTRER":
        rx = 0.50 + 0.08 * smooth_t
        ry = 0.52 - 0.06 * smooth_t
        return AvatarPoseState((0.35, 0.72), (rx, ry), "rest", "index")

    if sign_name == "INTERPRETE":
        angle = 2.0 * np.pi * t
        lx = 0.44 + 0.06 * np.cos(angle)
        ly = 0.52 + 0.06 * np.sin(angle)
        rx = 0.56 + 0.06 * np.cos(angle + np.pi)
        ry = 0.52 + 0.06 * np.sin(angle + np.pi)
        return AvatarPoseState((lx, ly), (rx, ry), "open", "open")

    if sign_name == "ECOLE":
        # Deux mains devant le buste qui battent légèrement ensemble
        offset = 0.04 * pulse
        return AvatarPoseState((0.44 + offset, 0.52), (0.56 - offset, 0.52), "open", "open")

    if sign_name == "FAMILLE":
        # Deux mains décrivant un cercle horizontal englobant devant le torse
        arc = np.pi * smooth_t
        lx = 0.48 - 0.12 * np.sin(arc)
        rx = 0.52 + 0.12 * np.sin(arc)
        ly = 0.48 + 0.08 * (1.0 - np.cos(arc))
        return AvatarPoseState((lx, ly), (rx, ly), "open", "open")

    if sign_name == "AMI":
        # Deux mains fermées qui se rejoignent chaleureusement au centre
        lx = 0.38 + 0.09 * smooth_t
        rx = 0.62 - 0.09 * smooth_t
        return AvatarPoseState((lx, 0.54), (rx, 0.54), "fist", "fist")

    return AvatarPoseState((0.36, 0.70), (0.64, 0.70), "open", "open")


def interpolate_avatar_poses(
    start_pose: AvatarPoseState,
    end_pose: AvatarPoseState,
    blend: float,
) -> AvatarPoseState:
    """Interpole avec lissage cosinus deux états cinématiques (transition inter-signes v3)."""
    alpha = float(0.5 - 0.5 * np.cos(np.pi * np.clip(blend, 0.0, 1.0)))
    lx = start_pose.left_wrist[0] * (1.0 - alpha) + end_pose.left_wrist[0] * alpha
    ly = start_pose.left_wrist[1] * (1.0 - alpha) + end_pose.left_wrist[1] * alpha
    rx = start_pose.right_wrist[0] * (1.0 - alpha) + end_pose.right_wrist[0] * alpha
    ry = start_pose.right_wrist[1] * (1.0 - alpha) + end_pose.right_wrist[1] * alpha
    l_shape = start_pose.left_hand_shape if alpha < 0.5 else end_pose.left_hand_shape
    r_shape = start_pose.right_hand_shape if alpha < 0.5 else end_pose.right_hand_shape
    return AvatarPoseState((lx, ly), (rx, ry), l_shape, r_shape)


def compute_sequence_avatar_pose(
    signs: list[str],
    sequence_progress: float,
) -> tuple[str, AvatarPoseState]:
    """Calcule l'état de l'avatar pour une phrase complète de signes enchaînés (Text-to-Sign v3)."""
    if not signs:
        return "REPOS", compute_canonical_sign_pose("REPOS", 0.0)

    clamped = float(np.clip(sequence_progress, 0.0, 0.9999))
    scaled = clamped * len(signs)
    idx = int(scaled)
    local_t = scaled - idx
    current_sign = signs[idx]

    # Les derniers 20% du cycle d'un signe servent à la transition fluide vers le suivant
    if local_t <= 0.80 or len(signs) == 1:
        sign_t = min(1.0, local_t / 0.80) if len(signs) > 1 else local_t
        return current_sign, compute_canonical_sign_pose(current_sign, sign_t)

    next_sign = signs[(idx + 1) % len(signs)]
    blend = (local_t - 0.80) / 0.20
    pose_end = compute_canonical_sign_pose(current_sign, 1.0)
    pose_next = compute_canonical_sign_pose(next_sign, 0.0)
    return current_sign, interpolate_avatar_poses(pose_end, pose_next, blend)


def _draw_styled_hand(
    canvas: np.ndarray,
    wrist_px: tuple[int, int],
    hand_spec: tuple[str, bool],
) -> None:
    """Dessine une main anatomique stylisée selon la configuration gestuelle et la latéralité."""
    hand_shape, is_left = hand_spec
    wx, wy = wrist_px
    direction = -1 if is_left else 1
    hand_color = (255, 220, 90) if is_left else (90, 245, 175)

    if hand_shape == "rest":
        cv2.circle(canvas, (wx, wy), 7, (80, 90, 105), -1, cv2.LINE_AA)
        return

    if hand_shape == "fist":
        cv2.circle(canvas, (wx, wy), 12, (45, 55, 70), -1, cv2.LINE_AA)
        cv2.circle(canvas, (wx, wy), 12, hand_color, 2, cv2.LINE_AA)
        for k_offset in (-7, -2, 3, 8):
            cv2.circle(canvas, (wx + k_offset, wy - 8), 3, (60, 255, 255), -1, cv2.LINE_AA)
        return

    if hand_shape == "l_shape":
        # Configuration en 'L' (PAPA, COMPRENDRE) : index vertical + pouce horizontal
        cv2.circle(canvas, (wx, wy), 10, (45, 55, 70), -1, cv2.LINE_AA)
        cv2.circle(canvas, (wx, wy), 10, hand_color, 2, cv2.LINE_AA)
        index_tip = (wx, wy - 34)
        thumb_tip = (wx - direction * 24, wy - 2)
        cv2.line(canvas, (wx, wy - 6), index_tip, hand_color, 4, cv2.LINE_AA)
        cv2.line(canvas, (wx, wy), thumb_tip, hand_color, 4, cv2.LINE_AA)
        cv2.circle(canvas, index_tip, 4, (60, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, thumb_tip, 4, (60, 255, 255), -1, cv2.LINE_AA)
        return

    if hand_shape == "index":
        cv2.circle(canvas, (wx, wy), 9, (45, 55, 70), -1, cv2.LINE_AA)
        cv2.circle(canvas, (wx, wy), 9, hand_color, 2, cv2.LINE_AA)
        finger_tip = (wx, wy - 32)
        cv2.line(canvas, (wx, wy - 6), finger_tip, hand_color, 4, cv2.LINE_AA)
        cv2.circle(canvas, finger_tip, 4, (60, 255, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, (wx, wy), (wx + direction * 10, wy - 8), hand_color, 3, cv2.LINE_AA)
        return

    if hand_shape == "roof":
        tip_x = wx + (18 if is_left else -18)
        tip_y = wy - 24
        cv2.line(canvas, (wx, wy), (tip_x, tip_y), hand_color, 4, cv2.LINE_AA)
        cv2.circle(canvas, (tip_x, tip_y), 4, (60, 255, 255), -1, cv2.LINE_AA)
        return

    # Main ouverte standard (open) : paume + 5 doigts déployés
    cv2.circle(canvas, (wx, wy), 10, (40, 50, 65), -1, cv2.LINE_AA)
    cv2.circle(canvas, (wx, wy), 10, hand_color, 2, cv2.LINE_AA)
    finger_angles = [-35, -15, 0, 15, 35] if not is_left else [35, 15, 0, -15, -35]
    for angle_deg in finger_angles:
        rad = np.radians(angle_deg)
        fx = int(wx + 24 * np.sin(rad))
        fy = int(wy - 24 * np.cos(rad))
        cv2.line(canvas, (wx, wy), (fx, fy), hand_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (fx, fy), 3, (60, 255, 255), -1, cv2.LINE_AA)


def render_parametric_pose_state(canvas: np.ndarray, pose: AvatarPoseState) -> None:
    """Dessine l'avatar anatomique sur le canvas à partir d'un état cinématique donné."""
    height, width = canvas.shape[:2]
    center_x = width // 2
    head_center = (center_x, int(height * 0.22))
    head_radius = 42

    neck_top = (center_x, int(height * 0.22 + head_radius))
    neck_base = (center_x, int(height * 0.35))
    shoulder_l = (center_x - 100, int(height * 0.36))
    shoulder_r = (center_x + 100, int(height * 0.36))
    waist_l = (center_x - 70, int(height * 0.72))
    waist_r = (center_x + 70, int(height * 0.72))

    torso_polygon = np.array([shoulder_l, shoulder_r, waist_r, waist_l], dtype=np.int32)
    cv2.fillPoly(canvas, [torso_polygon], (30, 34, 42), lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [torso_polygon], True, (65, 75, 90), 2, lineType=cv2.LINE_AA)

    cv2.line(canvas, neck_top, neck_base, (160, 175, 195), 4, cv2.LINE_AA)
    cv2.circle(canvas, head_center, head_radius, (38, 44, 55), -1, cv2.LINE_AA)
    cv2.circle(canvas, head_center, head_radius, (170, 190, 215), 2, cv2.LINE_AA)
    cv2.circle(canvas, (head_center[0] - 16, head_center[1] - 6), 4, (240, 245, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (head_center[0] + 16, head_center[1] - 6), 4, (240, 245, 255), -1, cv2.LINE_AA)

    cv2.line(canvas, shoulder_l, neck_base, (85, 100, 120), 2, cv2.LINE_AA)
    cv2.line(canvas, shoulder_r, neck_base, (85, 100, 120), 2, cv2.LINE_AA)

    lw_px = (int(pose.left_wrist[0] * width), int(pose.left_wrist[1] * height))
    rw_px = (int(pose.right_wrist[0] * width), int(pose.right_wrist[1] * height))

    elbow_l = ((shoulder_l[0] + lw_px[0]) // 2 - 28, max(shoulder_l[1] + 25, (shoulder_l[1] + lw_px[1]) // 2 + 15))
    elbow_r = ((shoulder_r[0] + rw_px[0]) // 2 + 28, max(shoulder_r[1] + 25, (shoulder_r[1] + rw_px[1]) // 2 + 15))

    for shoulder, elbow, wrist in [(shoulder_l, elbow_l, lw_px), (shoulder_r, elbow_r, rw_px)]:
        cv2.line(canvas, shoulder, elbow, (80, 180, 245), 4, cv2.LINE_AA)
        cv2.line(canvas, elbow, wrist, (80, 180, 245), 4, cv2.LINE_AA)
        cv2.circle(canvas, elbow, 6, (255, 220, 100), -1, cv2.LINE_AA)

    _draw_styled_hand(canvas, lw_px, (pose.left_hand_shape, True))
    _draw_styled_hand(canvas, rw_px, (pose.right_hand_shape, False))


def render_parametric_avatar_frame(
    canvas: np.ndarray,
    sign_name: str,
    progress: float,
) -> None:
    """Génère le rendu 3D volumétrique du mannequin humain rigué (V3) pour un signe."""
    rig = get_sign_rig_at_progress(sign_name, progress)
    render_rig_mannequin_cv(canvas, rig, yaw_deg=0.0)
