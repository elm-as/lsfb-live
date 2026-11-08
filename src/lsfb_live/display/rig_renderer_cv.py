"""Moteur de rendu 3D volumétrique du mannequin humain rigué sous OpenCV (V3).

Projette un RigSkeleton3D avec rotation orbitale (Yaw), tri de profondeur Z (algorithme
du peintre), capsules cylindriques ombrées (Lambert + spéculaire) et mains articulées
à 21 phalanges 3D avec paume polygonale.
"""

import cv2
import numpy as np

from lsfb_live.avatar.rig_skeleton import RigSkeleton3D
from lsfb_live.display.visualizer import FINGERTIP_INDICES, HAND_CONNECTIONS


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

    # Perspective douce (focale f = 2.8)
    cam_dist = 2.8 + rz
    persp = 2.8 / max(cam_dist, 0.8)

    scale = width * 0.34 * persp
    cx = width * 0.50
    cy = height * 0.45

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

    dark_bgr = (max(0, int(b * 0.48)), max(0, int(g * 0.48)), max(0, int(r * 0.48)))
    spec_bgr = (min(255, int(b * 1.35 + 25)), min(255, int(g * 1.35 + 25)), min(255, int(r * 1.35 + 25)))

    cv2.line(canvas, pt1, pt2, dark_bgr, radius * 2 + 2, cv2.LINE_AA)
    cv2.line(canvas, pt1, pt2, base_bgr, max(2, radius * 2 - 2), cv2.LINE_AA)
    cv2.line(canvas, pt1, pt2, spec_bgr, max(1, radius // 2), cv2.LINE_AA)


def _draw_articulated_hand_3d(
    canvas: np.ndarray,
    hand_3d: np.ndarray,
    transform_spec: tuple[float, tuple[int, int], bool],
) -> None:
    """Dessine la paume polygonale 3D et les 20 segments phalangiens d'une main riguée."""
    yaw_rad, viewport, is_left = transform_spec
    pts_2d = [_rotate_and_project(hand_3d[i], yaw_rad, viewport)[:2] for i in range(21)]

    # 1. Paume polygonale remplie (Poignet 0 -> Index 5 -> Majeur 9 -> Annulaire 13 -> Auriculaire 17)
    palm_poly = np.array([pts_2d[0], pts_2d[5], pts_2d[9], pts_2d[13], pts_2d[17]], dtype=np.int32)
    palm_fill = (65, 55, 42) if is_left else (45, 68, 52)
    bone_color = (245, 200, 75) if is_left else (85, 240, 165)

    cv2.fillPoly(canvas, [palm_poly], palm_fill, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [palm_poly], True, bone_color, 2, lineType=cv2.LINE_AA)

    # 2. Phalanges articulées des 5 doigts
    for idx1, idx2 in HAND_CONNECTIONS:
        thickness = 3 if idx1 in (0, 1, 5, 9, 13, 17) else 2
        cv2.line(canvas, pts_2d[idx1], pts_2d[idx2], bone_color, thickness, cv2.LINE_AA)

    # 3. Articulations inter-phalangiennes et pulpes des doigts
    for j_idx in range(21):
        is_tip = j_idx in FINGERTIP_INDICES
        rad = 4 if is_tip else 2
        j_color = (70, 255, 255) if is_tip else (240, 245, 250)
        cv2.circle(canvas, pts_2d[j_idx], rad, j_color, -1, cv2.LINE_AA)


def render_rig_mannequin_cv(
    canvas: np.ndarray,
    rig: RigSkeleton3D,
    yaw_deg: float = 0.0,
) -> None:
    """Génère le rendu complet du mannequin humain 3D rigué avec tri de profondeur Z."""
    height, width = canvas.shape[:2]
    viewport = (width, height)
    yaw_rad = float(np.radians(yaw_deg))

    # Grille de repère spatial discret en arrière-plan
    cx = width // 2
    cv2.line(canvas, (cx, 110), (cx, height - 65), (30, 35, 44), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, int(height * 0.79)), (150, 26), 0, 0, 360, (36, 42, 54), 1, cv2.LINE_AA)

    # Projection des repères anatomiques du tronc et de la tête
    ls_px = _rotate_and_project(rig.left_shoulder, yaw_rad, viewport)[:2]
    rs_px = _rotate_and_project(rig.right_shoulder, yaw_rad, viewport)[:2]
    neck_px = _rotate_and_project(rig.neck, yaw_rad, viewport)[:2]
    pelvis_px = _rotate_and_project(rig.pelvis, yaw_rad, viewport)[:2]
    head_px = _rotate_and_project(rig.head_center, yaw_rad, viewport)[:2]
    nose_px = _rotate_and_project(rig.nose_tip, yaw_rad, viewport)[:2]

    hip_l = (pelvis_px[0] - int(width * 0.10), pelvis_px[1])
    hip_r = (pelvis_px[0] + int(width * 0.10), pelvis_px[1])

    # 1. Rendu volumétrique du buste anatomique (trapèze thoracique + plastron pectoral)
    torso_poly = np.array([ls_px, rs_px, hip_r, hip_l], dtype=np.int32)
    cv2.fillPoly(canvas, [torso_poly], (34, 40, 50), lineType=cv2.LINE_AA)

    chest_mid = ((ls_px[0] + rs_px[0]) // 2, (ls_px[1] + rs_px[1]) // 2 + 42)
    chest_poly = np.array([ls_px, rs_px, (hip_r[0] + 8, chest_mid[1]), (hip_l[0] - 8, chest_mid[1])], dtype=np.int32)
    cv2.fillPoly(canvas, [chest_poly], (44, 52, 66), lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [torso_poly], True, (85, 100, 122), 2, lineType=cv2.LINE_AA)

    # Colonne sternale et clavicules
    cv2.line(canvas, neck_px, pelvis_px, (58, 68, 84), 2, cv2.LINE_AA)
    _draw_shaded_capsule(canvas, (neck_px, ls_px), (5, (90, 108, 130)))
    _draw_shaded_capsule(canvas, (neck_px, rs_px), (5, (90, 108, 130)))

    # 2. Cou et Tête sculptée 3D avec orientation du regard
    _draw_shaded_capsule(canvas, (neck_px, head_px), (9, (115, 130, 155)))
    head_rx = max(28, int(width * 0.068))
    head_ry = max(34, int(width * 0.082))
    cv2.ellipse(canvas, head_px, (head_rx, head_ry), 0, 0, 360, (46, 54, 68), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, head_px, (head_rx, head_ry), 0, 0, 360, (165, 185, 210), 2, cv2.LINE_AA)

    # Yeux et arête nasale orientés en 3D
    gaze_dx = int((nose_px[0] - head_px[0]) * 0.65)
    eye_y = head_px[1] - 6
    cv2.circle(canvas, (head_px[0] - 14 + gaze_dx, eye_y), 4, (235, 245, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (head_px[0] + 14 + gaze_dx, eye_y), 4, (235, 245, 255), -1, cv2.LINE_AA)
    cv2.line(canvas, (head_px[0] + gaze_dx, eye_y), nose_px, (180, 200, 225), 2, cv2.LINE_AA)

    # 3. Tri de profondeur Z des membres supérieurs (Algorithme du Peintre)
    le_proj = _rotate_and_project(rig.left_elbow, yaw_rad, viewport)
    re_proj = _rotate_and_project(rig.right_elbow, yaw_rad, viewport)
    lw_proj = _rotate_and_project(rig.left_wrist, yaw_rad, viewport)
    rw_proj = _rotate_and_project(rig.right_wrist, yaw_rad, viewport)

    limbs = [
        (le_proj[2], "limb", (ls_px, le_proj[:2]), (11, (75, 145, 215)), le_proj[:2]),
        (re_proj[2], "limb", (rs_px, re_proj[:2]), (11, (75, 145, 215)), re_proj[:2]),
        (lw_proj[2], "limb", (le_proj[:2], lw_proj[:2]), (9, (90, 175, 245)), lw_proj[:2]),
        (rw_proj[2], "limb", (re_proj[:2], rw_proj[:2]), (9, (90, 175, 245)), rw_proj[:2]),
        (lw_proj[2] - 0.05, "hand_l", rig.left_hand, None, None),
        (rw_proj[2] - 0.05, "hand_r", rig.right_hand, None, None),
    ]
    # Du plus éloigné (Z grand) au plus proche de la caméra (Z négatif)
    limbs.sort(key=lambda item: item[0], reverse=True)

    for _, kind, data, style, joint_px in limbs:
        if kind == "limb":
            _draw_shaded_capsule(canvas, data, style)
            cv2.circle(canvas, joint_px, 7, (255, 220, 110), -1, cv2.LINE_AA)
        elif kind == "hand_l":
            _draw_articulated_hand_3d(canvas, data, (yaw_rad, viewport, True))
        elif kind == "hand_r":
            _draw_articulated_hand_3d(canvas, data, (yaw_rad, viewport, False))
