"""Modèle mathématique du squelette humain rigué 3D et Cinématique Inverse (IK).

Convertit des repères normalisés (65, 3) ou des repères webcam en un rig humain 3D
anatomiquement contraint (longueurs d'os constantes, IK 2-segments bras/avant-bras
et mains articulées à 21 phalanges 3D).
"""

from dataclasses import dataclass
import numpy as np

from lsfb_live.capture.holistic_extractor import FrameLandmarks
from lsfb_live.landmarks.normalizer import normalize_spatial_coordinates, pack_raw_landmarks

# Proportions anatomiques canoniques (en unités de largeur bi-acromiale inter-épaules = 1.0)
UPPER_ARM_LENGTH = 0.68
FOREARM_LENGTH = 0.64
CANONICAL_PALM_SIZE = 0.25


@dataclass(frozen=True)
class RigSkeleton3D:
    """État 3D complet d'un mannequin humain rigué dans le repère centré-épaules.

    Axe X : gauche (-) vers droite (+) de l'écran (vue miroir face à l'utilisateur).
    Axe Y : haut (-) vers bas (+).
    Axe Z : vers l'avant/caméra (-) vers l'arrière (+).
    """

    head_center: np.ndarray
    nose_tip: np.ndarray
    neck: np.ndarray
    pelvis: np.ndarray
    left_shoulder: np.ndarray
    right_shoulder: np.ndarray
    left_elbow: np.ndarray
    right_elbow: np.ndarray
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray


def solve_two_bone_arm_ik(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    pole_dir: np.ndarray,
) -> np.ndarray:
    """Résout analytiquement la position 3D du coude par Cinématique Inverse (IK à 2 os).

    Garantit que le bras et l'avant-bras conservent une longueur anatomique réaliste
    et que le coude plie naturellement selon le vecteur pôle anatomique.
    """
    diff = wrist - shoulder
    dist = float(np.linalg.norm(diff))
    max_reach = UPPER_ARM_LENGTH + FOREARM_LENGTH - 0.02

    if dist < 1e-4:
        return shoulder + np.array([0.0, UPPER_ARM_LENGTH, 0.10], dtype=np.float32)

    forward = diff / dist
    clamped_dist = min(dist, max_reach)

    # Loi des cosinus dans le triangle (Épaule - Coude - Poignet)
    cos_angle = (
        UPPER_ARM_LENGTH**2 + clamped_dist**2 - FOREARM_LENGTH**2
    ) / max(2.0 * UPPER_ARM_LENGTH * clamped_dist, 1e-4)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    sin_angle = float(np.sqrt(max(0.0, 1.0 - cos_angle**2)))

    # Orthogonalisation de Gram-Schmidt pour l'axe de flexion du coude
    bend_axis = pole_dir - np.dot(pole_dir, forward) * forward
    bend_norm = float(np.linalg.norm(bend_axis))
    if bend_norm < 1e-4:
        bend_axis = np.array([0.0, 1.0, 0.2], dtype=np.float32)
    else:
        bend_axis = bend_axis / bend_norm

    elbow = shoulder + UPPER_ARM_LENGTH * (cos_angle * forward + sin_angle * bend_axis)
    return elbow.astype(np.float32)


def build_relaxed_hand_3d(wrist: np.ndarray, is_mirror_left: bool) -> np.ndarray:
    """Génère une main 3D anatomique à 21 articulations au repos lorsque la main est inactive."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = wrist
    side = -1.0 if is_mirror_left else 1.0

    finger_angles_deg = [-32.0, -14.0, 0.0, 13.0, 28.0]
    finger_lengths = [0.18, 0.24, 0.26, 0.24, 0.20]

    for f_idx, (ang, tot_len) in enumerate(zip(finger_angles_deg, finger_lengths)):
        rad = np.radians(ang * side)
        dx = float(np.sin(rad))
        dy = float(np.cos(rad))
        base_joint = 1 + f_idx * 4
        for phalanx in range(4):
            frac = (phalanx + 1) / 4.0
            curl_z = -0.04 * (frac**2)
            hand[base_joint + phalanx] = wrist + np.array(
                [dx * tot_len * frac, dy * tot_len * frac, curl_z],
                dtype=np.float32,
            )
    return hand


def _anchor_and_scale_hand(
    raw_hand: np.ndarray,
    target_wrist: np.ndarray,
    is_mirror_left: bool,
) -> np.ndarray:
    """Ancre les 21 phalanges sur le poignet 3D et normalise l'échelle palmaire."""
    local = raw_hand - raw_hand[0:1, :]
    palm_len = float(np.linalg.norm(local[9]))
    if palm_len < 0.04 or np.abs(raw_hand[0, 0]) > 1.25:
        return build_relaxed_hand_3d(target_wrist, is_mirror_left)

    scale = CANONICAL_PALM_SIZE / palm_len
    scaled_local = local * scale
    # Inversion miroir X pour que le rig face à l'utilisateur agisse comme un vrai miroir
    scaled_local[:, 0] = -scaled_local[:, 0]
    scaled_local[:, 2] = np.clip(scaled_local[:, 2], -0.25, 0.25)
    return (target_wrist[np.newaxis, :] + scaled_local).astype(np.float32)


def retarget_normalized_65_to_rig(landmarks_65_3: np.ndarray) -> RigSkeleton3D:
    """Convertit un squelette normalisé (65, 3) en RigSkeleton3D anatomique en vue miroir."""
    pts = landmarks_65_3.reshape(65, 3).astype(np.float32)

    # Coordonnées canoniques stables du buste et de la tête avec micro-expressivité
    nose_raw = np.array([-pts[0, 0], pts[0, 1], np.clip(pts[0, 2] * 0.2, -0.2, 0.2)], dtype=np.float32)
    head_y = float(np.clip(nose_raw[1], -0.72, -0.45))
    head_x = float(np.clip(nose_raw[0], -0.15, 0.15))
    head_center = np.array([head_x, head_y, -0.05], dtype=np.float32)
    nose_tip = head_center + np.array([0.0, 0.03, -0.16], dtype=np.float32)

    neck = np.array([head_x * 0.4, -0.14, 0.0], dtype=np.float32)
    pelvis = np.array([0.0, 0.92, 0.04], dtype=np.float32)

    # En vue miroir, l'épaule gauche de l'écran (-0.5) correspond à la main gauche du dataset
    left_shoulder = np.array([-0.50, 0.0, 0.0], dtype=np.float32)
    right_shoulder = np.array([0.50, 0.0, 0.0], dtype=np.float32)

    lh_raw = pts[23:44]
    rh_raw = pts[44:65]

    def extract_mirror_wrist(hand_pts: np.ndarray, default_x: float) -> np.ndarray:
        palm = float(np.linalg.norm(hand_pts[9] - hand_pts[0]))
        if palm < 0.04 or abs(float(hand_pts[0, 0])) > 1.20 or abs(float(hand_pts[0, 1])) > 1.35:
            return np.array([default_x, 0.78, -0.08], dtype=np.float32)
        wx = float(np.clip(-hand_pts[0, 0], -0.95, 0.95))
        wy = float(np.clip(hand_pts[0, 1], -0.75, 0.95))
        wz = float(np.clip(-0.18 + hand_pts[0, 2] * 0.15, -0.45, 0.10))
        return np.array([wx, wy, wz], dtype=np.float32)

    # En miroir : la main droite anatomique (rh_raw) est à droite de l'écran (+X) ou vice-versa
    # Dans le dataset LSFB (vue caméra), left_hand (23:44) a X > 0, donc après -X elle est à gauche (-X) !
    left_wrist = extract_mirror_wrist(lh_raw, -0.42)
    right_wrist = extract_mirror_wrist(rh_raw, 0.42)

    pole_left = np.array([-0.35, 0.85, 0.25], dtype=np.float32)
    pole_right = np.array([0.35, 0.85, 0.25], dtype=np.float32)
    left_elbow = solve_two_bone_arm_ik(left_shoulder, left_wrist, pole_left)
    right_elbow = solve_two_bone_arm_ik(right_shoulder, right_wrist, pole_right)

    left_hand = _anchor_and_scale_hand(lh_raw, left_wrist, is_mirror_left=True)
    right_hand = _anchor_and_scale_hand(rh_raw, right_wrist, is_mirror_left=False)

    return RigSkeleton3D(
        head_center=head_center,
        nose_tip=nose_tip,
        neck=neck,
        pelvis=pelvis,
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        left_elbow=left_elbow,
        right_elbow=right_elbow,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        left_hand=left_hand,
        right_hand=right_hand,
    )


def retarget_frame_landmarks_to_rig(landmarks: FrameLandmarks) -> RigSkeleton3D:
    """Retargeting temps réel des repères webcam vers le RigSkeleton3D."""
    packed = pack_raw_landmarks(
        landmarks.pose[np.newaxis, ...],
        landmarks.left_hand[np.newaxis, ...],
        landmarks.right_hand[np.newaxis, ...],
    )
    normalized = normalize_spatial_coordinates(packed)[0]
    return retarget_normalized_65_to_rig(normalized)


def interpolate_rig_skeletons(
    rig_a: RigSkeleton3D,
    rig_b: RigSkeleton3D,
    blend: float,
) -> RigSkeleton3D:
    """Interpole en 3D deux états du rig humain avec lissage cosinus et IK recalculée."""
    alpha = float(0.5 - 0.5 * np.cos(np.pi * np.clip(blend, 0.0, 1.0)))

    def lerp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((1.0 - alpha) * a + alpha * b).astype(np.float32)

    lw = lerp(rig_a.left_wrist, rig_b.left_wrist)
    rw = lerp(rig_a.right_wrist, rig_b.right_wrist)
    ls = lerp(rig_a.left_shoulder, rig_b.left_shoulder)
    rs = lerp(rig_a.right_shoulder, rig_b.right_shoulder)

    le = solve_two_bone_arm_ik(ls, lw, np.array([-0.35, 0.85, 0.25], dtype=np.float32))
    re = solve_two_bone_arm_ik(rs, rw, np.array([0.35, 0.85, 0.25], dtype=np.float32))

    return RigSkeleton3D(
        head_center=lerp(rig_a.head_center, rig_b.head_center),
        nose_tip=lerp(rig_a.nose_tip, rig_b.nose_tip),
        neck=lerp(rig_a.neck, rig_b.neck),
        pelvis=lerp(rig_a.pelvis, rig_b.pelvis),
        left_shoulder=ls,
        right_shoulder=rs,
        left_elbow=le,
        right_elbow=re,
        left_wrist=lw,
        right_wrist=rw,
        left_hand=lerp(rig_a.left_hand, rig_b.left_hand),
        right_hand=lerp(rig_a.right_hand, rig_b.right_hand),
    )
