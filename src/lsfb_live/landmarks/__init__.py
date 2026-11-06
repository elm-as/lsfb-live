"""Sélection des points et normalisation des poses, partagées entre entraînement et live."""

from lsfb_live.landmarks.constants import (
    BISTREAM_FEATURE_DIM,
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
from lsfb_live.landmarks.phonological_features import (
    PHONOLOGICAL_FEATURE_DIM,
    assemble_bistream_features,
    extract_phonological_features,
)

__all__ = [
    "BISTREAM_FEATURE_DIM",
    "COORDINATE_DIM",
    "ENRICHED_FEATURE_DIM",
    "FEATURE_DIM",
    "HAND_LANDMARK_COUNT",
    "LEFT_SHOULDER_INDEX",
    "PHONOLOGICAL_FEATURE_DIM",
    "RIGHT_SHOULDER_INDEX",
    "TARGET_SEQUENCE_LENGTH",
    "TOTAL_LANDMARKS",
    "UPPER_POSE_LANDMARK_COUNT",
    "assemble_bistream_features",
    "compute_instantaneous_velocities",
    "extract_phonological_features",
    "normalize_spatial_coordinates",
    "pack_raw_landmarks",
    "resample_temporal_sequence",
    "transform_enriched_landmarks_pipeline",
    "transform_landmarks_pipeline",
]
