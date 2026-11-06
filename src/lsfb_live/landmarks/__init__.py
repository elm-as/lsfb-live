"""Sélection des points et normalisation des poses, partagées entre entraînement et live."""

from lsfb_live.landmarks.constants import (
    COORDINATE_DIM,
    ENRICHED_FEATURE_DIM,
    FEATURE_DIM,
    HAND_LANDMARK_COUNT,
    LEFT_SHOULDER_INDEX,
    RIGHT_SHOULDER_INDEX,
    TARGET_SEQUENCE_LENGTH,
    TOTAL_LANDMARKS,
    UPPER_POSE_LANDMARK_COUNT,
)
from lsfb_live.landmarks.normalizer import (
    compute_instantaneous_velocities,
    normalize_spatial_coordinates,
    pack_raw_landmarks,
    resample_temporal_sequence,
    transform_enriched_landmarks_pipeline,
    transform_landmarks_pipeline,
)

__all__ = [
    "COORDINATE_DIM",
    "ENRICHED_FEATURE_DIM",
    "FEATURE_DIM",
    "HAND_LANDMARK_COUNT",
    "LEFT_SHOULDER_INDEX",
    "RIGHT_SHOULDER_INDEX",
    "TARGET_SEQUENCE_LENGTH",
    "TOTAL_LANDMARKS",
    "UPPER_POSE_LANDMARK_COUNT",
    "compute_instantaneous_velocities",
    "normalize_spatial_coordinates",
    "pack_raw_landmarks",
    "resample_temporal_sequence",
    "transform_enriched_landmarks_pipeline",
    "transform_landmarks_pipeline",
]
