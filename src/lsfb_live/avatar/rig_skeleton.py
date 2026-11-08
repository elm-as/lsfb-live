"""Modèle mathématique du squelette humain rigué 3D et Cinématique Inverse (IK).

Convertit des repères normalisés (65, 3) ou des repères webcam en un rig humain 3D
anatomiquement contraint (longueurs d'os constantes, IK 2-segments bras/avant-bras
et mains articulées à 21 phalanges 3D fidèles au modèle de reconnaissance).
"""

from dataclasses import dataclass
import numpy as np

from lsfb_live.capture.holistic_extractor import FrameLandmarks
from lsfb_live.landmarks.normalizer import normalize_spatial_coordinates, pack_raw_landmarks

UPPER_ARM_LENGTH = 0.68
FOREARM_LENGTH = 0.64

# Offsets anatomiques canoniques des repères crâniens (0..10 : nez, yeux, oreilles, bouche)
_CANONICAL_FACE_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0568, -0.1607, 0.1256],
        [0.1018, -0.1655, 0.1231],
        [0.1422, -0.1699, 0.1227],
        [-0.0721, -0.1559, 0.1364],
        [-0.1130, -0.1563, 0.1331],
        [-0.1515, -0.1571, 0.1308],
        [0.1955, -0.1295, 0.9820],
        [-0.2038, -0.1132, 1.0274],
        [0.0800, 0.1289, 0.3590],
        [-0.0682, 0.1278, 0.3534],
    ],
    dtype=np.float32,
)


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
    left_hand_active: bool = True
    right_hand_active: bool = True


def solve_two_bone_arm_ik(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    pole_dir: np.ndarray,
) -> np.ndarray:
    """Résout analytiquement la position 3D du coude par Cinématique Inverse (IK à 2 os)."""
    diff = wrist - shoulder
    dist = float(np.linalg.norm(diff))
    max_reach = UPPER_ARM_LENGTH + FOREARM_LENGTH - 0.02

    if dist < 1e-4:
        return shoulder + np.array([0.0, UPPER_ARM_LENGTH, 0.10], dtype=np.float32)

    forward = diff / dist
    clamped_dist = min(dist, max_reach)

    cos_angle = (
        UPPER_ARM_LENGTH**2 + clamped_dist**2 - FOREARM_LENGTH**2
    ) / max(2.0 * UPPER_ARM_LENGTH * clamped_dist, 1e-4)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    sin_angle = float(np.sqrt(max(0.0, 1.0 - cos_angle**2)))

    bend_axis = pole_dir - np.dot(pole_dir, forward) * forward
    bend_norm = float(np.linalg.norm(bend_axis))
    if bend_norm < 1e-4:
        bend_axis = np.array([0.0, 1.0, 0.2], dtype=np.float32)
    else:
        bend_axis = bend_axis / bend_norm

    elbow = shoulder + UPPER_ARM_LENGTH * (cos_angle * forward + sin_angle * bend_axis)
    return elbow.astype(np.float32)


def build_relaxed_hand_3d(wrist: np.ndarray, is_mirror_left: bool) -> np.ndarray:
    """Génère une main 3D anatomique à 21 articulations au repos le long du corps."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = wrist
    side = -1.0 if is_mirror_left else 1.0

    finger_angles_deg = [-28.0, -12.0, 0.0, 11.0, 24.0]
    finger_lengths = [0.16, 0.22, 0.24, 0.22, 0.18]

    for f_idx, (ang, tot_len) in enumerate(zip(finger_angles_deg, finger_lengths)):
        rad = np.radians(ang * side)
        dx = float(np.sin(rad))
        dy = float(np.cos(rad))
        base_joint = 1 + f_idx * 4
        for phalanx in range(4):
            frac = (phalanx + 1) / 4.0
            curl_z = -0.03 * (frac**2)
            hand[base_joint + phalanx] = wrist + np.array(
                [dx * tot_len * frac, dy * tot_len * frac, curl_z],
                dtype=np.float32,
            )
    return hand


def _extract_mirror_hand_and_wrist(
    raw_hand: np.ndarray,
    default_x: float,
    is_mirror_left: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extrait le poignet et les 21 phalanges 3D en préservant la géométrie exacte du signe."""
    local = raw_hand - raw_hand[0:1, :]
    palm_len = float(np.linalg.norm(local[9]))
    is_valid = (palm_len > 0.04) and (abs(float(raw_hand[0, 0])) < 1.25) and (abs(float(raw_hand[0, 1])) < 1.45)

    if not is_valid:
        rest_wrist = np.array([default_x, 0.86, 0.02], dtype=np.float32)
        return rest_wrist, build_relaxed_hand_3d(rest_wrist, is_mirror_left), False

    # Coordonnées exactes (X, Y) du poignet en vue miroir (-X) et profondeur anatomique Z = -0.18 devant le buste
    wrist_z = float(np.clip(-0.18 + raw_hand[0, 2] * 0.04, -0.28, -0.08))
    wrist = np.array([-raw_hand[0, 0], raw_hand[0, 1], wrist_z], dtype=np.float32)

    # Préservation fidèle des vecteurs phalangiens 3D locaux (inversion miroir sur l'axe X uniquement)
    mirrored_local = local.copy()
    mirrored_local[:, 0] = -mirrored_local[:, 0]
    hand_3d = (wrist[np.newaxis, :] + mirrored_local).astype(np.float32)
    return wrist, hand_3d, True


def retarget_normalized_65_to_rig(landmarks_65_3: np.ndarray) -> RigSkeleton3D:
    """Convertit un squelette normalisé (65, 3) en RigSkeleton3D fidèle au modèle."""
    pts = landmarks_65_3.reshape(65, 3).astype(np.float32)

    head_x = float(np.clip(-pts[0, 0], -0.25, 0.25))
    head_y = float(np.clip(pts[0, 1], -0.85, -0.35))
    head_center = np.array([head_x, head_y, -0.05], dtype=np.float32)
    nose_tip = head_center + np.array([0.0, 0.03, -0.16], dtype=np.float32)

    neck = np.array([head_x * 0.35, -0.14, 0.0], dtype=np.float32)
    pelvis = np.array([0.0, 0.92, 0.04], dtype=np.float32)

    left_shoulder = np.array([-pts[11, 0], pts[11, 1], 0.0], dtype=np.float32)
    right_shoulder = np.array([-pts[12, 0], pts[12, 1], 0.0], dtype=np.float32)
    if float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])) < 0.2:
        left_shoulder = np.array([-0.50, 0.0, 0.0], dtype=np.float32)
        right_shoulder = np.array([0.50, 0.0, 0.0], dtype=np.float32)

    left_wrist, left_hand, lh_active = _extract_mirror_hand_and_wrist(pts[23:44], -0.44, True)
    right_wrist, right_hand, rh_active = _extract_mirror_hand_and_wrist(pts[44:65], 0.44, False)

    pole_left = np.array([-0.38, 0.85, 0.25], dtype=np.float32)
    pole_right = np.array([0.38, 0.85, 0.25], dtype=np.float32)
    left_elbow = solve_two_bone_arm_ik(left_shoulder, left_wrist, pole_left)
    right_elbow = solve_two_bone_arm_ik(right_shoulder, right_wrist, pole_right)

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
        left_hand_active=lh_active,
        right_hand_active=rh_active,
    )


def rig_to_normalized_65(rig: RigSkeleton3D) -> np.ndarray:
    """Opération inverse : reconstruit le tenseur normalisé (65, 3) depuis le RigSkeleton3D."""
    pts = np.zeros((65, 3), dtype=np.float32)
    nose_unmirrored = np.array([-rig.head_center[0], rig.head_center[1], -1.5048], dtype=np.float32)
    pts[:11] = nose_unmirrored[np.newaxis, :] + _CANONICAL_FACE_OFFSETS
    pts[11] = [-rig.left_shoulder[0], rig.left_shoulder[1], -0.1075]
    pts[12] = [-rig.right_shoulder[0], rig.right_shoulder[1], 0.1075]
    pts[13] = [-rig.left_elbow[0], rig.left_elbow[1], rig.left_elbow[2]]
    pts[14] = [-rig.right_elbow[0], rig.right_elbow[1], rig.right_elbow[2]]
    pts[15] = [-rig.left_wrist[0], rig.left_wrist[1], 0.0]
    pts[16] = [-rig.right_wrist[0], rig.right_wrist[1], 0.0]

    if rig.left_hand_active:
        lh = rig.left_hand.copy()
        lh[:, 0] = -lh[:, 0]
        pts[23:44] = lh
    if rig.right_hand_active:
        rh = rig.right_hand.copy()
        rh[:, 0] = -rh[:, 0]
        pts[44:65] = rh
    return pts


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

    le = solve_two_bone_arm_ik(ls, lw, np.array([-0.38, 0.85, 0.25], dtype=np.float32))
    re = solve_two_bone_arm_ik(rs, rw, np.array([0.38, 0.85, 0.25], dtype=np.float32))

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
        left_hand_active=rig_a.left_hand_active if alpha < 0.5 else rig_b.left_hand_active,
        right_hand_active=rig_a.right_hand_active if alpha < 0.5 else rig_b.right_hand_active,
    )
