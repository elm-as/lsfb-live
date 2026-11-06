"""Extraction des descripteurs géométriques et phonologiques pour la LSFB."""

import numpy as np

PHONOLOGICAL_FEATURE_DIM = 26


def compute_inter_hand_features(landmarks_40_65_3: np.ndarray) -> np.ndarray:
    """Calcule la distance et le vecteur relatif entre poignets et index (5 valeurs).

    Args:
        landmarks_40_65_3: Tenseur de repères normalisés (T, 65, 3).

    Returns:
        Tableau float32 de forme (T, 5).
    """
    lw = landmarks_40_65_3[:, 15, :]  # Poignet gauche
    rw = landmarks_40_65_3[:, 16, :]  # Poignet droit

    # Distance euclidienne inter-poignets et vecteur relatif (4 valeurs)
    wrist_disp = rw - lw
    wrist_dist = np.linalg.norm(wrist_disp, axis=-1, keepdims=True)

    # Contact entre pointes des index gauche (31) et droit (52) (1 valeur)
    l_index = landmarks_40_65_3[:, 31, :]
    r_index = landmarks_40_65_3[:, 52, :]
    index_dist = np.linalg.norm(r_index - l_index, axis=-1, keepdims=True)

    return np.concatenate([wrist_dist, wrist_disp, index_dist], axis=-1).astype(np.float32)


def compute_body_anchor_features(landmarks_40_65_3: np.ndarray) -> np.ndarray:
    """Calcule les distances des mains aux ancrages corporels (nez, torse, hauteur) (6 valeurs).

    Args:
        landmarks_40_65_3: Tenseur de repères normalisés (T, 65, 3).

    Returns:
        Tableau float32 de forme (T, 6).
    """
    nose = landmarks_40_65_3[:, 0, :]
    lw = landmarks_40_65_3[:, 15, :]
    rw = landmarks_40_65_3[:, 16, :]

    chest = (landmarks_40_65_3[:, 11, :] + landmarks_40_65_3[:, 12, :]) / 2.0
    shoulder_y = chest[:, 1:2]

    # Distances aux zones d'articulation phonologiques
    dist_lw_nose = np.linalg.norm(lw - nose, axis=-1, keepdims=True)
    dist_rw_nose = np.linalg.norm(rw - nose, axis=-1, keepdims=True)
    dist_lw_chest = np.linalg.norm(lw - chest, axis=-1, keepdims=True)
    dist_rw_chest = np.linalg.norm(rw - chest, axis=-1, keepdims=True)

    # Élévation relative par rapport à la ligne des épaules
    elev_lw = lw[:, 1:2] - shoulder_y
    elev_rw = rw[:, 1:2] - shoulder_y

    return np.concatenate([dist_lw_nose, dist_rw_nose, dist_lw_chest, dist_rw_chest, elev_lw, elev_rw], axis=-1).astype(np.float32)


def compute_finger_extension_features(landmarks_40_65_3: np.ndarray) -> np.ndarray:
    """Calcule les 10 extensions des doigts par rapport aux paumes (5 gauche, 5 droite).

    Args:
        landmarks_40_65_3: Tenseur de repères normalisés (T, 65, 3).

    Returns:
        Tableau float32 de forme (T, 10).
    """
    # Main gauche (poignet 23, bouts: 27=pouce, 31=index, 35=majeur, 39=annulaire, 43=auriculaire)
    lw_base = landmarks_40_65_3[:, 23, :]
    lh_tips = [27, 31, 35, 39, 43]
    lh_extensions = [np.linalg.norm(landmarks_40_65_3[:, tip, :] - lw_base, axis=-1, keepdims=True) for tip in lh_tips]

    # Main droite (poignet 44, bouts: 48=pouce, 52=index, 56=majeur, 60=annulaire, 64=auriculaire)
    rw_base = landmarks_40_65_3[:, 44, :]
    rh_tips = [48, 52, 56, 60, 64]
    rh_extensions = [np.linalg.norm(landmarks_40_65_3[:, tip, :] - rw_base, axis=-1, keepdims=True) for tip in rh_tips]

    return np.concatenate(lh_extensions + rh_extensions, axis=-1).astype(np.float32)


def compute_kinematic_scalars(landmarks_40_65_3: np.ndarray) -> np.ndarray:
    """Calcule les grandeurs dynamiques scalaires : vitesse, accélération, divergence (5 valeurs).

    Args:
        landmarks_40_65_3: Tenseur de repères normalisés (T, 65, 3).

    Returns:
        Tableau float32 de forme (T, 5).
    """
    total_frames = landmarks_40_65_3.shape[0]
    lw = landmarks_40_65_3[:, 15, :]
    rw = landmarks_40_65_3[:, 16, :]

    # Vitesses instantanées scalaires des poignets
    lw_vel = np.zeros((total_frames, 1), dtype=np.float32)
    rw_vel = np.zeros((total_frames, 1), dtype=np.float32)
    lw_vel[1:] = np.linalg.norm(lw[1:] - lw[:-1], axis=-1, keepdims=True)
    rw_vel[1:] = np.linalg.norm(rw[1:] - rw[:-1], axis=-1, keepdims=True)

    # Accélérations instantanées
    lw_acc = np.zeros((total_frames, 1), dtype=np.float32)
    rw_acc = np.zeros((total_frames, 1), dtype=np.float32)
    lw_acc[1:] = np.abs(lw_vel[1:] - lw_vel[:-1])
    rw_acc[1:] = np.abs(rw_vel[1:] - rw_vel[:-1])

    # Taux de rapprochement/éloignement des mains
    inter_dist = np.linalg.norm(rw - lw, axis=-1, keepdims=True)
    inter_speed = np.zeros((total_frames, 1), dtype=np.float32)
    inter_speed[1:] = inter_dist[1:] - inter_dist[:-1]

    return np.concatenate([lw_vel, rw_vel, lw_acc, rw_acc, inter_speed], axis=-1).astype(np.float32)


def extract_phonological_features(normalized_sequence: np.ndarray) -> np.ndarray:
    """Extrait l'ensemble des 26 descripteurs phonologiques invariants sur la séquence.

    Args:
        normalized_sequence: Tableau de forme (T, 195) ou (T, 65, 3).

    Returns:
        Tableau float32 de forme (T, 26).
    """
    if normalized_sequence.ndim == 2:
        reshaped = normalized_sequence.reshape(-1, 65, 3)
    else:
        reshaped = normalized_sequence

    inter_hand = compute_inter_hand_features(reshaped)          # 5
    anchors = compute_body_anchor_features(reshaped)             # 6
    extensions = compute_finger_extension_features(reshaped)     # 10
    kinematics = compute_kinematic_scalars(reshaped)             # 5

    return np.concatenate([inter_hand, anchors, extensions, kinematics], axis=-1).astype(np.float32)


def assemble_bistream_features(positions: np.ndarray) -> np.ndarray:
    """Structure les repères en 3 flux orthogonaux : corps (138), mains (252), phonologie (26).

    Args:
        positions: Séquence temporelle de positions normalisées de forme (T, 195).

    Returns:
        Tableau float32 de forme (T, 416) découplé pour le BiStreamMSSTClassifier.
    """
    if positions.shape[-1] != 195:
        raise ValueError(f"Attendu 195 caractéristiques de position, reçu {positions.shape[-1]}")

    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]

    # Découpage anatomique : 23 repères supérieurs (69 features), 42 repères mains (126 features)
    body_pos = positions[:, :69]
    body_vel = velocities[:, :69]
    hands_pos = positions[:, 69:195]
    hands_vel = velocities[:, 69:195]

    phono = extract_phonological_features(positions)

    return np.concatenate([body_pos, body_vel, hands_pos, hands_vel, phono], axis=-1).astype(np.float32)

