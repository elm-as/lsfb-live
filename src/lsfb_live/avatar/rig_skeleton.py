"""Modèle mathématique du squelette humain rigué 3D, FK phalangienne fixe et IK 2-os.

Convertit des repères normalisés (65, 3) ou des repères webcam en un rig humain 3D
strictement contraint anatomiquement : longueurs d'os constantes pour les bras et les
20 segments phalangiens (zéro étirement des doigts), coudes naturels par IK 2-segments.
"""

from dataclasses import dataclass
import numpy as np

from lsfb_live.capture.holistic_extractor import FrameLandmarks
from lsfb_live.landmarks.normalizer import normalize_spatial_coordinates, pack_raw_landmarks

# Proportions anthropométriques canoniques (largeur bi-acromiale = 1.0)
UPPER_ARM_LENGTH = 0.72
FOREARM_LENGTH = 0.68

# Chaînes hiérarchiques (parent -> enfant) et longueurs fixes des 20 os de la main
_HAND_FK_CHAINS: tuple[tuple[tuple[int, int, float], ...], ...] = (
    ((0, 1, 0.068), (1, 2, 0.056), (2, 3, 0.044), (3, 4, 0.036)),      # Pouce
    ((0, 5, 0.148), (5, 6, 0.062), (6, 7, 0.038), (7, 8, 0.030)),      # Index
    ((0, 9, 0.152), (9, 10, 0.068), (10, 11, 0.042), (11, 12, 0.032)), # Majeur
    ((0, 13, 0.142), (13, 14, 0.062), (14, 15, 0.038), (15, 16, 0.030)), # Annulaire
    ((0, 17, 0.128), (17, 18, 0.048), (18, 19, 0.030), (19, 20, 0.025)), # Auriculaire
)

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
    max_reach = UPPER_ARM_LENGTH + FOREARM_LENGTH - 0.03

    if dist < 1e-4:
        return shoulder + np.array([0.0, UPPER_ARM_LENGTH, 0.12], dtype=np.float32)

    forward = diff / dist
    clamped_dist = float(np.clip(dist, 0.20, max_reach))

    cos_angle = (
        UPPER_ARM_LENGTH**2 + clamped_dist**2 - FOREARM_LENGTH**2
    ) / max(2.0 * UPPER_ARM_LENGTH * clamped_dist, 1e-4)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    sin_angle = float(np.sqrt(max(0.0, 1.0 - cos_angle**2)))

    # Projection orthogonale du vecteur pôle anatomique (vers le bas, l'extérieur et l'arrière)
    bend_axis = pole_dir - np.dot(pole_dir, forward) * forward
    bend_norm = float(np.linalg.norm(bend_axis))
    if bend_norm < 1e-3:
        bend_axis = np.array([float(np.sign(pole_dir[0])) or 1.0, 0.5, 0.5], dtype=np.float32)
        bend_axis = bend_axis / float(np.linalg.norm(bend_axis))
    else:
        bend_axis = bend_axis / bend_norm

    elbow = shoulder + UPPER_ARM_LENGTH * (cos_angle * forward + sin_angle * bend_axis)
    return elbow.astype(np.float32)


def enforce_anatomical_hand_fk(wrist: np.ndarray, mirrored_hand: np.ndarray) -> np.ndarray:
    """Reconstruit la main 3D par cinématique directe (FK) à longueurs d'os strictement fixes.

    Préserve l'orientation 3D exacte de chaque phalange tout en empêchant tout étirement
    ou déformation de perspective des doigts.
    """
    constrained = np.zeros((21, 3), dtype=np.float32)
    constrained[0] = wrist

    for chain in _HAND_FK_CHAINS:
        for parent_idx, child_idx, bone_len in chain:
            raw_vec = mirrored_hand[child_idx] - mirrored_hand[parent_idx]
            norm = float(np.linalg.norm(raw_vec))
            if norm < 1e-5:
                unit_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            else:
                unit_dir = raw_vec / norm
            constrained[child_idx] = constrained[parent_idx] + unit_dir * bone_len

    return constrained


def build_relaxed_hand_3d(wrist: np.ndarray, is_mirror_left: bool) -> np.ndarray:
    """Génère une main 3D anatomique à 21 articulations au repos le long de la hanche."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = wrist
    side = -1.0 if is_mirror_left else 1.0
    finger_angles = [-20.0, -7.0, 0.0, 7.0, 15.0]

    for chain, ang_deg in zip(_HAND_FK_CHAINS, finger_angles):
        rad = np.radians(ang_deg * side)
        base_dir = np.array([np.sin(rad) * 0.35, np.cos(rad), -0.18], dtype=np.float32)
        base_dir /= float(np.linalg.norm(base_dir))
        for step, (parent_idx, child_idx, bone_len) in enumerate(chain):
            curl_dir = base_dir + np.array([-side * 0.12 * step, 0.0, -0.18 * step], dtype=np.float32)
            curl_dir /= float(np.linalg.norm(curl_dir))
            hand[child_idx] = hand[parent_idx] + curl_dir * bone_len
    return hand


def _extract_mirror_hand_and_wrist(
    raw_hand: np.ndarray,
    default_x: float,
    is_mirror_left: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extrait le poignet et applique le verrouillage FK des 20 os de la main."""
    local = raw_hand - raw_hand[0:1, :]
    palm_len = float(np.linalg.norm(local[9]))
    is_valid = (
        (palm_len > 0.04)
        and (abs(float(raw_hand[0, 0])) < 1.25)
        and (abs(float(raw_hand[0, 1])) < 1.35)
    )

    if not is_valid:
        rest_wrist = np.array([default_x, 1.12, 0.02], dtype=np.float32)
        return rest_wrist, build_relaxed_hand_3d(rest_wrist, is_mirror_left), False

    wrist_z = float(np.clip(-0.26 + raw_hand[0, 2] * 0.05, -0.38, -0.12))
    wrist = np.array([-raw_hand[0, 0], raw_hand[0, 1], wrist_z], dtype=np.float32)

    mirrored_raw = local.copy()
    mirrored_raw[:, 0] = -mirrored_raw[:, 0]
    hand_3d = enforce_anatomical_hand_fk(wrist, mirrored_raw)
    return wrist, hand_3d, True


def retarget_normalized_65_to_rig(landmarks_65_3: np.ndarray) -> RigSkeleton3D:
    """Convertit un squelette normalisé (65, 3) en RigSkeleton3D anatomiquement contraint."""
    pts = landmarks_65_3.reshape(65, 3).astype(np.float32)

    head_x = float(np.clip(-pts[0, 0], -0.18, 0.18))
    head_y = float(np.clip(pts[0, 1], -0.78, -0.48))
    head_center = np.array([head_x, head_y, -0.04], dtype=np.float32)
    nose_tip = head_center + np.array([0.0, 0.03, -0.16], dtype=np.float32)

    neck = np.array([head_x * 0.35, -0.14, 0.0], dtype=np.float32)
    pelvis = np.array([0.0, 0.96, 0.04], dtype=np.float32)

    left_shoulder = np.array([-0.48, 0.0, 0.0], dtype=np.float32)
    right_shoulder = np.array([0.48, 0.0, 0.0], dtype=np.float32)

    left_wrist, left_hand, lh_active = _extract_mirror_hand_and_wrist(pts[23:44], -0.56, True)
    right_wrist, right_hand, rh_active = _extract_mirror_hand_and_wrist(pts[44:65], 0.56, False)

    pole_left = np.array([-0.55, 1.0, 0.38], dtype=np.float32)
    pole_right = np.array([0.55, 1.0, 0.38], dtype=np.float32)
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
    """Interpole en 3D deux états du rig humain avec lissage cosinus, FK et IK recalculées."""
    alpha = float(0.5 - 0.5 * np.cos(np.pi * np.clip(blend, 0.0, 1.0)))

    def lerp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((1.0 - alpha) * a + alpha * b).astype(np.float32)

    lw = lerp(rig_a.left_wrist, rig_b.left_wrist)
    rw = lerp(rig_a.right_wrist, rig_b.right_wrist)
    ls = lerp(rig_a.left_shoulder, rig_b.left_shoulder)
    rs = lerp(rig_a.right_shoulder, rig_b.right_shoulder)

    le = solve_two_bone_arm_ik(ls, lw, np.array([-0.55, 1.0, 0.38], dtype=np.float32))
    re = solve_two_bone_arm_ik(rs, rw, np.array([0.55, 1.0, 0.38], dtype=np.float32))

    lh_blend = enforce_anatomical_hand_fk(lw, lerp(rig_a.left_hand, rig_b.left_hand))
    rh_blend = enforce_anatomical_hand_fk(rw, lerp(rig_a.right_hand, rig_b.right_hand))

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
        left_hand=lh_blend,
        right_hand=rh_blend,
        left_hand_active=rig_a.left_hand_active if alpha < 0.5 else rig_b.left_hand_active,
        right_hand_active=rig_a.right_hand_active if alpha < 0.5 else rig_b.right_hand_active,
    )
