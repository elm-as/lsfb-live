"""Tests unitaires pour la normalisation invariante des squelettes LSFB."""

import numpy as np

from lsfb_live.landmarks.constants import (
    ENRICHED_FEATURE_DIM,
    FEATURE_DIM,
    LEFT_SHOULDER_INDEX,
    RIGHT_SHOULDER_INDEX,
    TARGET_SEQUENCE_LENGTH,
    TOTAL_LANDMARKS,
)
from lsfb_live.landmarks.normalizer import (
    compute_instantaneous_velocities,
    normalize_spatial_coordinates,
    pack_raw_landmarks,
    resample_temporal_sequence,
    transform_enriched_landmarks_pipeline,
    transform_landmarks_pipeline,
)


def test_pack_raw_landmarks_concatenates_correctly_and_replaces_nan():
    """Vérifie l'empilement (F, 65, 3) et la substitution déterministe des NaN."""
    frames = 5
    pose = np.ones((frames, 33, 3), dtype=np.float32)
    pose[0, 0, 0] = np.nan
    lh = np.zeros((frames, 21, 3), dtype=np.float32)
    rh = np.full((frames, 21, 3), np.nan, dtype=np.float32)

    packed = pack_raw_landmarks(pose, lh, rh)

    assert packed.shape == (frames, TOTAL_LANDMARKS, 3)
    assert not np.isnan(packed).any()
    assert packed[0, 0, 0] == 0.0


def test_normalize_spatial_coordinates_centers_shoulders_and_scales_to_one():
    """Vérifie que le point médian des épaules est à l'origine et l'écart vaut 1.0."""
    frames = 3
    landmarks = np.zeros((frames, TOTAL_LANDMARKS, 3), dtype=np.float32)

    # Épaules écartées de 2.0 unités selon l'axe X (gauche: +1.0, droite: -1.0)
    landmarks[:, LEFT_SHOULDER_INDEX, 0] = 1.0
    landmarks[:, RIGHT_SHOULDER_INDEX, 0] = -1.0

    normalized = normalize_spatial_coordinates(landmarks)

    midpoint_x = (
        normalized[:, LEFT_SHOULDER_INDEX, 0] + normalized[:, RIGHT_SHOULDER_INDEX, 0]
    ) / 2.0
    assert np.allclose(midpoint_x, 0.0, atol=1e-5)

    shoulder_distance = np.linalg.norm(
        normalized[:, LEFT_SHOULDER_INDEX, :] - normalized[:, RIGHT_SHOULDER_INDEX, :],
        axis=1,
    )
    assert np.allclose(shoulder_distance, 1.0, atol=1e-5)


def test_resample_temporal_sequence_adjusts_length_smoothly():
    """Vérifie la transformation temporelle vers TARGET_SEQUENCE_LENGTH."""
    short_seq = np.linspace(0.0, 1.0, num=10).reshape(10, 1, 1).repeat(TOTAL_LANDMARKS, axis=1).repeat(3, axis=2)
    resampled = resample_temporal_sequence(short_seq, target_length=TARGET_SEQUENCE_LENGTH)

    assert resampled.shape == (TARGET_SEQUENCE_LENGTH, TOTAL_LANDMARKS, 3)
    assert np.isclose(resampled[0, 0, 0], 0.0, atol=1e-4)
    assert np.isclose(resampled[-1, 0, 0], 1.0, atol=1e-4)


def test_transform_landmarks_pipeline_outputs_correct_tensor():
    """Vérifie la forme finale (40, 195) de la chaîne complète."""
    frames = 15
    pose = np.random.randn(frames, 33, 3).astype(np.float32)
    lh = np.random.randn(frames, 21, 3).astype(np.float32)
    rh = np.random.randn(frames, 21, 3).astype(np.float32)

    features = transform_landmarks_pipeline(pose, lh, rh)

    assert features.shape == (TARGET_SEQUENCE_LENGTH, FEATURE_DIM)
    assert features.dtype == np.float32
    assert not np.isnan(features).any()


def test_compute_instantaneous_velocities_calculates_difference():
    """Vérifie le calcul différentiel de la vitesse le long de l'axe temporel."""
    positions = np.array([
        [0.0, 1.0],
        [2.0, 3.0],
        [5.0, 2.0],
    ], dtype=np.float32)

    velocities = compute_instantaneous_velocities(positions)

    assert velocities.shape == positions.shape
    assert np.allclose(velocities[0], [0.0, 0.0])
    assert np.allclose(velocities[1], [2.0, 2.0])
    assert np.allclose(velocities[2], [3.0, -1.0])


def test_transform_enriched_landmarks_pipeline_outputs_390_features():
    """Vérifie que le pipeline enrichi génère bien 390 features (195 pos + 195 vit)."""
    frames = 10
    pose = np.random.randn(frames, 33, 3).astype(np.float32)
    lh = np.random.randn(frames, 21, 3).astype(np.float32)
    rh = np.random.randn(frames, 21, 3).astype(np.float32)

    enriched = transform_enriched_landmarks_pipeline(pose, lh, rh)

    assert enriched.shape == (TARGET_SEQUENCE_LENGTH, ENRICHED_FEATURE_DIM)
    assert enriched.dtype == np.float32
    assert not np.isnan(enriched).any()


