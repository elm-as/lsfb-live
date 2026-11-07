"""Extraction des descripteurs géométriques, d'orientation palmaire et phonologiques LSFB."""

import numpy as np

PHONOLOGICAL_FEATURE_DIM = 26
EPSILON = 1e-5


def _extract_local_hand_and_wrist(
    hand_seq: np.ndarray,
    default_rest_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Extrait la forme 3D locale des doigts normalisée par la paume et la trajectoire du poignet.

    En centrant chaque main sur son propre poignet (indice 0), on élimine :
    1. Le décalage global du bras (95% de la variance parasite).
    2. La corruption de la coordonnée Z par l'axe Z des épaules.
    3. L'effet des mains absentes en webcam (qui s'annulent proprement à 0).
    """
    total_frames = hand_seq.shape[0]
    local_hand = hand_seq - hand_seq[:, 0:1, :]
    palm_size = np.linalg.norm(local_hand[:, 9:10, :], axis=-1, keepdims=True)
    valid_mask = palm_size[:, 0, 0] > 1e-4

    safe_palm = np.where(palm_size > 1e-4, palm_size, 1.0)
    unit_hand = np.where(palm_size > 1e-4, local_hand / safe_palm, 0.0).astype(np.float32)

    # Stockage du vecteur normal 3D de la paume dans le slot 0 (poignet local = 0)
    v_index = unit_hand[:, 5, :]
    v_pinky = unit_hand[:, 17, :]
    palm_normal = np.cross(v_index, v_pinky)
    normal_norm = np.linalg.norm(palm_normal, axis=-1, keepdims=True)
    unit_normal = np.where(normal_norm > EPSILON, palm_normal / np.maximum(normal_norm, EPSILON), 0.0)
    unit_hand[:, 0, :] = unit_normal.astype(np.float32)

    # Trajectoire globale du poignet dans le repère des épaules (Z mis à 0 pour éviter le bruit)
    true_wrist = np.zeros((total_frames, 3), dtype=np.float32)
    true_wrist[:, 0] = np.where(valid_mask, hand_seq[:, 0, 0], default_rest_xy[0])
    true_wrist[:, 1] = np.where(valid_mask, hand_seq[:, 0, 1], default_rest_xy[1])
    return unit_hand, true_wrist


def compute_inter_hand_features(
    lw: np.ndarray,
    rw: np.ndarray,
    landmarks_40_65_3: np.ndarray,
) -> np.ndarray:
    """Calcule la distance et le vecteur relatif entre vrais poignets et index (5 valeurs)."""
    wrist_disp = rw - lw
    wrist_dist = np.linalg.norm(wrist_disp, axis=-1, keepdims=True)

    # Contact entre pointes des index gauche (31) et droit (52) en plan XY épaules (1 valeur)
    l_index_xy = landmarks_40_65_3[:, 31, :2]
    r_index_xy = landmarks_40_65_3[:, 52, :2]
    index_dist = np.linalg.norm(r_index_xy - l_index_xy, axis=-1, keepdims=True)

    return np.concatenate([wrist_dist, wrist_disp, index_dist], axis=-1).astype(np.float32)


def compute_body_anchor_features(
    lw: np.ndarray,
    rw: np.ndarray,
    landmarks_40_65_3: np.ndarray,
) -> np.ndarray:
    """Calcule les distances des vrais poignets aux ancrages corporels (nez, torse, hauteur) (6 valeurs)."""
    nose_xy = landmarks_40_65_3[:, 0, :2]
    chest_xy = (landmarks_40_65_3[:, 11, :2] + landmarks_40_65_3[:, 12, :2]) / 2.0
    shoulder_y = chest_xy[:, 1:2]

    lw_xy = lw[:, :2]
    rw_xy = rw[:, :2]

    dist_lw_nose = np.linalg.norm(lw_xy - nose_xy, axis=-1, keepdims=True)
    dist_rw_nose = np.linalg.norm(rw_xy - nose_xy, axis=-1, keepdims=True)
    dist_lw_chest = np.linalg.norm(lw_xy - chest_xy, axis=-1, keepdims=True)
    dist_rw_chest = np.linalg.norm(rw_xy - chest_xy, axis=-1, keepdims=True)

    elev_lw = lw[:, 1:2] - shoulder_y
    elev_rw = rw[:, 1:2] - shoulder_y

    return np.concatenate(
        [dist_lw_nose, dist_rw_nose, dist_lw_chest, dist_rw_chest, elev_lw, elev_rw],
        axis=-1,
    ).astype(np.float32)


def compute_finger_extension_features(
    unit_lh: np.ndarray,
    unit_rh: np.ndarray,
) -> np.ndarray:
    """Calcule les 10 extensions des doigts normalisées par la taille de la paume (10 valeurs)."""
    tip_indices = [4, 8, 12, 16, 20]
    lh_extensions = [np.linalg.norm(unit_lh[:, tip, :], axis=-1, keepdims=True) for tip in tip_indices]
    rh_extensions = [np.linalg.norm(unit_rh[:, tip, :], axis=-1, keepdims=True) for tip in tip_indices]
    return np.concatenate(lh_extensions + rh_extensions, axis=-1).astype(np.float32)


def compute_kinematic_scalars(lw: np.ndarray, rw: np.ndarray) -> np.ndarray:
    """Calcule les grandeurs dynamiques scalaires : vitesse, accélération, divergence (5 valeurs)."""
    total_frames = lw.shape[0]

    lw_vel = np.zeros((total_frames, 1), dtype=np.float32)
    rw_vel = np.zeros((total_frames, 1), dtype=np.float32)
    lw_vel[1:] = np.linalg.norm(lw[1:] - lw[:-1], axis=-1, keepdims=True)
    rw_vel[1:] = np.linalg.norm(rw[1:] - rw[:-1], axis=-1, keepdims=True)

    lw_acc = np.zeros((total_frames, 1), dtype=np.float32)
    rw_acc = np.zeros((total_frames, 1), dtype=np.float32)
    lw_acc[1:] = np.abs(lw_vel[1:] - lw_vel[:-1])
    rw_acc[1:] = np.abs(rw_vel[1:] - rw_vel[:-1])

    inter_dist = np.linalg.norm(rw - lw, axis=-1, keepdims=True)
    inter_speed = np.zeros((total_frames, 1), dtype=np.float32)
    inter_speed[1:] = inter_dist[1:] - inter_dist[:-1]

    return np.concatenate([lw_vel, rw_vel, lw_acc, rw_acc, inter_speed], axis=-1).astype(np.float32)


def extract_phonological_features(normalized_sequence: np.ndarray) -> np.ndarray:
    """Extrait l'ensemble des 26 descripteurs phonologiques invariants sur la séquence."""
    reshaped = (
        normalized_sequence.reshape(-1, 65, 3)
        if normalized_sequence.ndim == 2
        else normalized_sequence
    )
    lh_seq = reshaped[:, 23:44, :]
    rh_seq = reshaped[:, 44:65, :]
    unit_lh, true_lw = _extract_local_hand_and_wrist(lh_seq, (0.36, 1.08))
    unit_rh, true_rw = _extract_local_hand_and_wrist(rh_seq, (-0.30, 1.08))

    inter_hand = compute_inter_hand_features(true_lw, true_rw, reshaped)      # 5
    anchors = compute_body_anchor_features(true_lw, true_rw, reshaped)        # 6
    extensions = compute_finger_extension_features(unit_lh, unit_rh)          # 10
    kinematics = compute_kinematic_scalars(true_lw, true_rw)                  # 5

    return np.concatenate([inter_hand, anchors, extensions, kinematics], axis=-1).astype(np.float32)


def assemble_bistream_features(positions: np.ndarray) -> np.ndarray:
    """Structure les repères en 3 flux : corps (138), mains centrées-poignet (252), phonologie (26).

    Args:
        positions: Séquence temporelle de positions normalisées de forme (T, 195).

    Returns:
        Tableau float32 de forme (T, 416) découplé pour le BiStreamMSSTClassifier.
    """
    if positions.shape[-1] != 195:
        raise ValueError(f"Attendu 195 caractéristiques de position, reçu {positions.shape[-1]}")

    total_frames = positions.shape[0]
    reshaped = positions.reshape(total_frames, 65, 3)

    pose_seq = reshaped[:, :23, :].copy()
    lh_seq = reshaped[:, 23:44, :]
    rh_seq = reshaped[:, 44:65, :]

    # 1. Extraction des formes de doigts 3D pures centrées sur leur propre poignet + normale paume
    unit_lh, true_lw = _extract_local_hand_and_wrist(lh_seq, (0.36, 1.08))
    unit_rh, true_rw = _extract_local_hand_and_wrist(rh_seq, (-0.30, 1.08))

    # 2. Ancrage des poignets du corps sur les vraies mains et atténuation du bruit Z monoculaire
    pose_seq[:, 15, :] = true_lw
    pose_seq[:, 16, :] = true_rw
    pose_seq[:, :, 2] *= 0.15

    body_pos = pose_seq.reshape(total_frames, 69)
    body_vel = np.zeros_like(body_pos)
    body_vel[1:] = body_pos[1:] - body_pos[:-1]

    hands_pos = np.concatenate(
        [unit_lh.reshape(total_frames, 63), unit_rh.reshape(total_frames, 63)],
        axis=-1,
    )
    hands_vel = np.zeros_like(hands_pos)
    hands_vel[1:] = hands_pos[1:] - hands_pos[:-1]

    phono = np.concatenate(
        [
            compute_inter_hand_features(true_lw, true_rw, reshaped),
            compute_body_anchor_features(true_lw, true_rw, reshaped),
            compute_finger_extension_features(unit_lh, unit_rh),
            compute_kinematic_scalars(true_lw, true_rw),
        ],
        axis=-1,
    ).astype(np.float32)

    return np.concatenate([body_pos, body_vel, hands_pos, hands_vel, phono], axis=-1).astype(np.float32)
