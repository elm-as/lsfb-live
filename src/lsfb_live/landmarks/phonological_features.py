"""Extraction des descripteurs géométriques, trajectoires multi-échelles et phonologie LSFB."""

import numpy as np

PHONOLOGICAL_FEATURE_DIM = 26
EPSILON = 1e-5


def _interpolate_missing_frames_1d(values: np.ndarray, valid_mask: np.ndarray, default_val: float) -> np.ndarray:
    """Interpole linéairement les trames manquantes entre trames valides sans saut au repos."""
    if not np.any(valid_mask):
        return np.full(len(values), default_val, dtype=np.float32)
    if np.all(valid_mask):
        return values.astype(np.float32)
    indices = np.arange(len(values))
    return np.interp(indices, indices[valid_mask], values[valid_mask]).astype(np.float32)


def _extract_local_hand_and_wrist(
    hand_seq: np.ndarray,
    default_rest_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Extrait la forme 3D locale des doigts centrée sur le poignet et la trajectoire continue.

    Élimine les artefacts d'interpolation aux frontières d'apparition de la main
    et maintient la continuité de la trajectoire du poignet sans saut parasite.
    """
    total_frames = hand_seq.shape[0]
    local_hand = hand_seq - hand_seq[:, 0:1, :]
    palm_size = np.linalg.norm(local_hand[:, 9:10, :], axis=-1, keepdims=True)
    palm_1d = palm_size[:, 0, 0]

    # Filtrage des trames fantômes issues de l'interpolation entre main absente (0) et main présente
    raw_valid = (palm_1d > 0.04) & (np.abs(hand_seq[:, 0, 0]) < 1.25) & (np.abs(hand_seq[:, 0, 1]) < 1.45)
    if np.any(raw_valid):
        median_palm = float(np.median(palm_1d[raw_valid]))
        valid_mask = raw_valid & (palm_1d >= 0.45 * median_palm)
    else:
        valid_mask = np.zeros(total_frames, dtype=bool)

    safe_palm = np.where(valid_mask[:, np.newaxis, np.newaxis], np.maximum(palm_size, 1e-4), 1.0)
    unit_hand = np.where(valid_mask[:, np.newaxis, np.newaxis], local_hand / safe_palm, 0.0).astype(np.float32)

    # Vecteur normal unitaire 3D de la paume stocké dans l'indice 0
    v_index = unit_hand[:, 5, :]
    v_pinky = unit_hand[:, 17, :]
    palm_normal = np.cross(v_index, v_pinky)
    normal_norm = np.linalg.norm(palm_normal, axis=-1, keepdims=True)
    unit_normal = np.where(normal_norm > EPSILON, palm_normal / np.maximum(normal_norm, EPSILON), 0.0)
    unit_hand[:, 0, :] = unit_normal.astype(np.float32)

    # Continuité temporelle des doigts si la main s'occulte brièvement pendant un mouvement rapide
    if np.any(valid_mask) and not np.all(valid_mask):
        flat_hand = unit_hand.reshape(total_frames, -1)
        for col in range(flat_hand.shape[1]):
            flat_hand[:, col] = _interpolate_missing_frames_1d(flat_hand[:, col], valid_mask, 0.0)
        unit_hand = flat_hand.reshape(total_frames, 21, 3)

    # Trajectoire continue du poignet sans saut vers la position de repos en cours de geste
    true_wrist = np.zeros((total_frames, 3), dtype=np.float32)
    true_wrist[:, 0] = _interpolate_missing_frames_1d(hand_seq[:, 0, 0], valid_mask, default_rest_xy[0])
    true_wrist[:, 1] = _interpolate_missing_frames_1d(hand_seq[:, 0, 1], valid_mask, default_rest_xy[1])
    return unit_hand, true_wrist


def compute_inter_hand_features(
    lw: np.ndarray,
    rw: np.ndarray,
    landmarks_40_65_3: np.ndarray,
) -> np.ndarray:
    """Calcule la distance et le vecteur relatif entre vrais poignets et index (5 valeurs)."""
    wrist_disp = rw - lw
    wrist_dist = np.linalg.norm(wrist_disp, axis=-1, keepdims=True)

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
    """Calcule les déplacements directionnels cumulés depuis le début du signe (5 valeurs).

    Encode explicitement la dynamique multi-frames (convergence RENCONTRER, divergence
    DIFFERENT, descente ADORER/CORPS, montée MAISON).
    """
    # Déplacement directionnel signé (dX, dY) de chaque main depuis le début de la séquence
    lw_disp_xy = lw[:, :2] - lw[0:1, :2]  # (T, 2)
    rw_disp_xy = rw[:, :2] - rw[0:1, :2]  # (T, 2)

    # Variation cumulée de l'écartement inter-mains depuis l'instant initial (convergence < 0, divergence > 0)
    inter_dist = np.linalg.norm(rw[:, :2] - lw[:, :2], axis=-1, keepdims=True)
    inter_delta = inter_dist - inter_dist[0:1, :]  # (T, 1)

    return np.concatenate([lw_disp_xy, rw_disp_xy, inter_delta], axis=-1).astype(np.float32)


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

    inter_hand = compute_inter_hand_features(true_lw, true_rw, reshaped)          # 5
    anchors = compute_body_anchor_features(true_lw, true_rw, reshaped)            # 6
    extensions = compute_finger_extension_features(unit_lh, unit_rh)              # 10
    kinematics = compute_kinematic_scalars(true_lw, true_rw)                      # 5

    return np.concatenate([inter_hand, anchors, extensions, kinematics], axis=-1).astype(np.float32)


def assemble_bistream_features(positions: np.ndarray) -> np.ndarray:
    """Structure les repères en 3 flux : corps + trajectoires (138), mains locales (252), phonologie (26)."""
    if positions.shape[-1] != 195:
        raise ValueError(f"Attendu 195 caractéristiques de position, reçu {positions.shape[-1]}")

    total_frames = positions.shape[0]
    reshaped = positions.reshape(total_frames, 65, 3)

    pose_seq = reshaped[:, :23, :].copy()
    lh_seq = reshaped[:, 23:44, :]
    rh_seq = reshaped[:, 44:65, :]

    # 1. Extraction des formes de doigts 3D pures et trajectoires continues des poignets
    unit_lh, true_lw = _extract_local_hand_and_wrist(lh_seq, (0.36, 1.08))
    unit_rh, true_rw = _extract_local_hand_and_wrist(rh_seq, (-0.30, 1.08))

    # 2. Remplacement des repères bras/mains bruités de Pose (13..22) par la cinématique exacte :
    # - 13, 14 : Coudes cinématiques reconstruits depuis épaules (11, 12) et vrais poignets
    pose_seq[:, 13, :] = 0.5 * (pose_seq[:, 11, :] + true_lw)
    pose_seq[:, 14, :] = 0.5 * (pose_seq[:, 12, :] + true_rw)
    # - 15, 16 : Vrais poignets gauche et droit
    pose_seq[:, 15, :] = true_lw
    pose_seq[:, 16, :] = true_rw
    # - 17, 18 : Déplacement cumulé des poignets depuis le début du signe (P(t) - P(0))
    pose_seq[:, 17, :] = true_lw - true_lw[0:1, :]
    pose_seq[:, 18, :] = true_rw - true_rw[0:1, :]
    # - 19, 20 : Déplacement à moyen terme sur fenêtre de 8 trames (P(t) - P(t-8))
    lag_idx = np.maximum(0, np.arange(total_frames) - 8)
    pose_seq[:, 19, :] = true_lw - true_lw[lag_idx, :]
    pose_seq[:, 20, :] = true_rw - true_rw[lag_idx, :]
    # - 21, 22 : Pointes des index dans le repère global des épaules
    pose_seq[:, 21, :] = true_lw + unit_lh[:, 8, :] * 0.22
    pose_seq[:, 22, :] = true_rw + unit_rh[:, 8, :] * 0.22
    pose_seq[:, :13, 2] *= 0.15

    body_pos = pose_seq.reshape(total_frames, 69)
    # Vitesse lissée sur 3 trames pour amplifier le vrai mouvement par rapport au bruit trame-à-trame
    vel_lag = np.maximum(0, np.arange(total_frames) - 3)
    body_vel = (body_pos - body_pos[vel_lag, :]).astype(np.float32)

    hands_pos = np.concatenate(
        [unit_lh.reshape(total_frames, 63), unit_rh.reshape(total_frames, 63)],
        axis=-1,
    )
    hands_vel = (hands_pos - hands_pos[vel_lag, :]).astype(np.float32)

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
