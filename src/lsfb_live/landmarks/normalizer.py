"""Pipeline pur et déterministe de normalisation des repères LSFB."""

import numpy as np

from lsfb_live.landmarks.constants import (
    COORDINATE_DIM,
    EPSILON_DISTANCE,
    FEATURE_DIM,
    HAND_LANDMARK_COUNT,
    LEFT_SHOULDER_INDEX,
    RIGHT_SHOULDER_INDEX,
    TARGET_SEQUENCE_LENGTH,
    TOTAL_LANDMARKS,
    UPPER_POSE_LANDMARK_COUNT,
)


def pack_raw_landmarks(
    pose: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> np.ndarray:
    """Concatène les repères bruts en un tenseur unifié (F, 65, 3) sans NaN.

    Args:
        pose: Séquence de repères du corps (F, 33, 3).
        left_hand: Séquence de la main gauche (F, 21, 3).
        right_hand: Séquence de la main droite (F, 21, 3).

    Returns:
        Tableau float32 de forme (F, 65, 3) débarrassé des NaN.
    """
    frame_count = max(len(pose), len(left_hand), len(right_hand))
    if frame_count == 0:
        return np.zeros((0, TOTAL_LANDMARKS, COORDINATE_DIM), dtype=np.float32)

    upper_pose = np.nan_to_num(pose[:frame_count, :UPPER_POSE_LANDMARK_COUNT, :COORDINATE_DIM], nan=0.0)
    lh_clean = np.nan_to_num(left_hand[:frame_count, :HAND_LANDMARK_COUNT, :COORDINATE_DIM], nan=0.0)
    rh_clean = np.nan_to_num(right_hand[:frame_count, :HAND_LANDMARK_COUNT, :COORDINATE_DIM], nan=0.0)

    return np.concatenate([upper_pose, lh_clean, rh_clean], axis=1).astype(np.float32)


def normalize_spatial_coordinates(landmarks: np.ndarray) -> np.ndarray:
    """Centre le squelette sur le milieu des épaules et met à l'échelle par la largeur des épaules.

    Args:
        landmarks: Tableau de forme (F, N, 3).

    Returns:
        Nouveau tableau normalisé invariant en translation et échelle.
    """
    if len(landmarks) == 0:
        return landmarks.copy()

    left_shoulder = landmarks[:, LEFT_SHOULDER_INDEX, :]
    right_shoulder = landmarks[:, RIGHT_SHOULDER_INDEX, :]

    midpoint = (left_shoulder + right_shoulder) / 2.0  # (F, 3)
    centered = landmarks - midpoint[:, np.newaxis, :]

    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)  # (F, 1)
    scale = np.where(shoulder_dist > EPSILON_DISTANCE, shoulder_dist, 1.0)  # (F, 1)

    normalized = centered / scale[:, np.newaxis, :]
    return normalized.astype(np.float32)


def resample_temporal_sequence(
    sequence: np.ndarray,
    target_length: int = TARGET_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Rééchantillonne uniformément une séquence temporelle par interpolation linéaire.

    Args:
        sequence: Tableau de forme (F, N, 3) ou (F, D).
        target_length: Nombre de frames cible (40 par défaut).

    Returns:
        Tableau de forme (target_length, N, 3) ou (target_length, D).
    """
    current_length = sequence.shape[0]
    if current_length == 0:
        target_shape = (target_length,) + sequence.shape[1:]
        return np.zeros(target_shape, dtype=np.float32)

    if current_length == target_length:
        return sequence.copy().astype(np.float32)

    if current_length == 1:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    original_times = np.linspace(0.0, 1.0, num=current_length)
    target_times = np.linspace(0.0, 1.0, num=target_length)

    flattened = sequence.reshape(current_length, -1)
    interpolated = np.empty((target_length, flattened.shape[1]), dtype=np.float32)

    for feature_idx in range(flattened.shape[1]):
        interpolated[:, feature_idx] = np.interp(
            target_times,
            original_times,
            flattened[:, feature_idx],
        )

    return interpolated.reshape((target_length,) + sequence.shape[1:])


def transform_landmarks_pipeline(
    pose: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> np.ndarray:
    """Orchestre la chaîne complète de transformation : empilement, centrage, échelle, rééchantillonnage.

    Args:
        pose: Repères bruts du corps.
        left_hand: Repères bruts de la main gauche.
        right_hand: Repères bruts de la main droite.

    Returns:
        Tableau aplati de forme (TARGET_SEQUENCE_LENGTH, FEATURE_DIM).
    """
    packed = pack_raw_landmarks(pose, left_hand, right_hand)
    if len(packed) == 0:
        return np.zeros((TARGET_SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

    spatially_normalized = normalize_spatial_coordinates(packed)
    temporally_resampled = resample_temporal_sequence(spatially_normalized, TARGET_SEQUENCE_LENGTH)

    return temporally_resampled.reshape(TARGET_SEQUENCE_LENGTH, FEATURE_DIM).astype(np.float32)

