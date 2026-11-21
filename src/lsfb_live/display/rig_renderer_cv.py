"""Moteur de rendu 3D volumétrique du mannequin humain rigué sous OpenCV (V3.1).

Projette un RigSkeleton3D (longueurs d'os phalangiens et humérales verrouillées)
avec rotation orbitale (Yaw), tri de profondeur Z (algorithme du peintre), capsules
anatomiques ombrées et mains volumétriques proportionnées.
"""

import cv2
import numpy as np

from lsfb_live.avatar.rig_skeleton import RigSkeleton3D
from lsfb_live.display.visualizer import HAND_CONNECTIONS

# Palette mannequin biomécanique sobre (BGR)
_TORSO_DARK = (38, 44, 54)
_TORSO_MID = (52, 60, 74)
_LIMB_SLATE = (155, 165, 178)
_JOINT_AMBER = (40, 158, 240)
_HAND_LEFT_FILL = (175, 195, 210)
_HAND_RIGHT_FILL = (185, 212, 200)
_HAND_REST_FILL = (100, 108, 120)


def _rotate_and_project(
    pt3d: np.ndarray,
    yaw_rad: float,
    viewport: tuple[int, int],
) -> tuple[int, int, float]:
    """Applique une rotation 3D d'angle lacet (Yaw) et projette en perspective 2D."""
    width, height = viewport
    x, y, z = float(pt3d[0]), float(pt3d[1]), float(pt3d[2])

    cos_y = float(np.cos(yaw_rad))
    sin_y = float(np.sin(yaw_rad))
    rx = x * cos_y - z * sin_y
    rz = x * sin_y + z * cos_y

    cam_dist = 3.1 + rz
    persp = 3.1 / max(cam_dist, 1.0)

    scale = width * 0.32 * persp
    cx = width * 0.50
    cy = height * 0.44

    px = int(np.clip(cx + rx * scale, 0, width - 1))
    py = int(np.clip(cy + y * scale, 0, height - 1))
    return px, py, rz


def _draw_shaded_capsule(
    canvas: np.ndarray,
    endpoints: tuple[tuple[int, int], tuple[int, int]],
    style: tuple[int, tuple[int, int, int]],
) -> None:
    """Trace un segment de membre 3D volumétrique avec ombrage cylindrique à 3 niveaux."""
    pt1, pt2 = endpoints
    radius, base_bgr = style
    b, g, r = base_bgr

    dark_bgr = (max(0, int(b * 0.52)), max(0, int(g * 0.52)), max(0, int(r * 0.52)))
    spec_bgr = (min(255, int(b * 1.25 + 18)), min(255, int(g * 1.25 + 18)), min(255, int(r * 1.25 + 18)))

    cv2.line(canvas, pt1, pt2, dark_bgr, radius * 2 + 2, cv2.LINE_AA)
    cv2.line(canvas, pt1, pt2, base_bgr, max(2, radius * 2 - 2), cv2.LINE_AA)
    cv2.line(canvas, pt1, pt2, spec_bgr, max(1, radius // 2), cv2.LINE_AA)


def _draw_articulated_hand_3d(
    canvas: np.ndarray,
    hand_3d: np.ndarray,
    transform_spec: tuple[float, tuple[int, int], bool, bool],
) -> None:
    """Dessine la main volumétrique 3D (paume pleine + 15 phalanges articulées proportionnées)."""
    yaw_rad, viewport, is_left, is_active = transform_spec
    pts_2d = [_rotate_and_project(hand_3d[i], yaw_rad, viewport)[:2] for i in range(21)]

    if not is_active:
        base_color = _HAND_REST_FILL
        outline = (80, 88, 100)
    else:
        base_color = _HAND_LEFT_FILL if is_left else _HAND_RIGHT_FILL
        outline = (210, 170, 70) if is_left else (90, 215, 145)

    # 1. Paume polygonale remplie et lissée (0 -> 1 -> 5 -> 9 -> 13 -> 17)
    palm_poly = np.array([pts_2d[0], pts_2d[1], pts_2d[5], pts_2d[9], pts_2d[13], pts_2d[17]], dtype=np.int32)
    cv2.fillPoly(canvas, [palm_poly], base_color, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [palm_poly], True, outline, 1, cv2.LINE_AA)

    # 2. Phalanges volumétriques des 5 doigts (épaisseur dégressive de la base à l'extrémité)
    for idx1, idx2 in HAND_CONNECTIONS:
        if idx1 in (5, 9, 13) and idx2 in (9, 13, 17):
            continue
        is_metacarpal = idx1 == 0
        thickness = 4 if is_metacarpal else (3 if idx1 in (1, 5, 9, 13, 17) else 2)
        cv2.line(canvas, pts_2d[idx1], pts_2d[idx2], outline, thickness + 2, cv2.LINE_AA)
        cv2.line(canvas, pts_2d[idx1], pts_2d[idx2], base_color, thickness, cv2.LINE_AA)

    # 3. Articulations discrètes des phalanges
    if is_active:
        for j_idx in (4, 8, 12, 16, 20):
            cv2.circle(canvas, pts_2d[j_idx], 2, outline, -1, cv2.LINE_AA)


def render_rig_mannequin_cv(
    canvas: np.ndarray,
    rig: RigSkeleton3D,
    yaw_deg: float = 0.0,
) -> None:
    """Génère le rendu complet du mannequin humain 3D rigué avec tri de profondeur Z."""
    height, width = canvas.shape[:2]
    viewport = (width, height)
    yaw_rad = float(np.radians(yaw_deg))

    cx = width // 2
    cv2.line(canvas, (cx, 105), (cx, height - 60), (28, 32, 40), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, int(height * 0.80)), (135, 22), 0, 0, 360, (34, 40, 50), 1, cv2.LINE_AA)

    ls_px = _rotate_and_project(rig.left_shoulder, yaw_rad, viewport)[:2]
    rs_px = _rotate_and_project(rig.right_shoulder, yaw_rad, viewport)[:2]
    neck_px = _rotate_and_project(rig.neck, yaw_rad, viewport)[:2]
    pelvis_px = _rotate_and_project(rig.pelvis, yaw_rad, viewport)[:2]
    head_px = _rotate_and_project(rig.head_center, yaw_rad, viewport)[:2]
    nose_px = _rotate_and_project(rig.nose_tip, yaw_rad, viewport)[:2]

    hip_l = (pelvis_px[0] - int(width * 0.095), pelvis_px[1])
    hip_r = (pelvis_px[0] + int(width * 0.095), pelvis_px[1])

    # 1. Buste anatomique sculpté
    torso_poly = np.array([ls_px, rs_px, hip_r, hip_l], dtype=np.int32)
    cv2.fillPoly(canvas, [torso_poly], _TORSO_DARK, lineType=cv2.LINE_AA)
    chest_y = (ls_px[1] + pelvis_px[1]) // 2 - 12
    chest_poly = np.array([ls_px, rs_px, (hip_r[0] + 10, chest_y), (hip_l[0] - 10, chest_y)], dtype=np.int32)
    cv2.fillPoly(canvas, [chest_poly], _TORSO_MID, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [torso_poly], True, (90, 102, 120), 2, cv2.LINE_AA)

    # Épaules deltoïdes et clavicules
    _draw_shaded_capsule(canvas, (neck_px, ls_px), (6, (105, 118, 134)))
    _draw_shaded_capsule(canvas, (neck_px, rs_px), (6, (105, 118, 134)))
    cv2.circle(canvas, ls_px, 11, _JOINT_AMBER, -1, cv2.LINE_AA)
    cv2.circle(canvas, rs_px, 11, _JOINT_AMBER, -1, cv2.LINE_AA)

    # 2. Cou et Tête anatomique proportionnée
    _draw_shaded_capsule(canvas, (neck_px, head_px), (10, (140, 152, 168)))
    head_rx = max(24, int(width * 0.062))
    head_ry = max(30, int(width * 0.078))
    cv2.ellipse(canvas, head_px, (head_rx, head_ry), 0, 0, 360, (168, 178, 192), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, head_px, (head_rx, head_ry), 0, 0, 360, (95, 108, 126), 2, cv2.LINE_AA)

    gaze_dx = int((nose_px[0] - head_px[0]) * 0.55)
    eye_y = head_px[1] - 4
    cv2.line(canvas, (head_px[0] - 15 + gaze_dx, eye_y), (head_px[0] - 6 + gaze_dx, eye_y), (40, 48, 60), 2, cv2.LINE_AA)
    cv2.line(canvas, (head_px[0] + 6 + gaze_dx, eye_y), (head_px[0] + 15 + gaze_dx, eye_y), (40, 48, 60), 2, cv2.LINE_AA)
    cv2.line(canvas, (head_px[0] + gaze_dx, eye_y + 2), nose_px, (90, 102, 118), 2, cv2.LINE_AA)

    # 3. Tri de profondeur Z des bras, avant-bras et mains volumétriques
    le_proj = _rotate_and_project(rig.left_elbow, yaw_rad, viewport)
    re_proj = _rotate_and_project(rig.right_elbow, yaw_rad, viewport)
    lw_proj = _rotate_and_project(rig.left_wrist, yaw_rad, viewport)
    rw_proj = _rotate_and_project(rig.right_wrist, yaw_rad, viewport)

    limbs = [
        (le_proj[2], "limb", (ls_px, le_proj[:2]), (10, _LIMB_SLATE), le_proj[:2]),
        (re_proj[2], "limb", (rs_px, re_proj[:2]), (10, _LIMB_SLATE), re_proj[:2]),
        (lw_proj[2], "limb", (le_proj[:2], lw_proj[:2]), (8, (175, 185, 198)), lw_proj[:2]),
        (rw_proj[2], "limb", (re_proj[:2], rw_proj[:2]), (8, (175, 185, 198)), rw_proj[:2]),
        (lw_proj[2] - 0.05, "hand_l", rig.left_hand, None, None),
        (rw_proj[2] - 0.05, "hand_r", rig.right_hand, None, None),
    ]
    limbs.sort(key=lambda item: item[0], reverse=True)

    for _, kind, data, style, joint_px in limbs:
        if kind == "limb":
            _draw_shaded_capsule(canvas, data, style)
            cv2.circle(canvas, joint_px, 6, _JOINT_AMBER, -1, cv2.LINE_AA)
        elif kind == "hand_l":
            _draw_articulated_hand_3d(canvas, data, (yaw_rad, viewport, True, rig.left_hand_active))
        elif kind == "hand_r":
            _draw_articulated_hand_3d(canvas, data, (yaw_rad, viewport, False, rig.right_hand_active))
