"""Tests unitaires pour les descripteurs phonologiques explicites."""

import numpy as np
from lsfb_live.landmarks.phonological_features import (
    PHONOLOGICAL_FEATURE_DIM,
    assemble_bistream_features,
    extract_phonological_features,
)


def test_extract_phonological_features_shape_and_finite():
    """Vérifie que l'extraction produit un tenseur (40, 26) sans NaN ni infini."""
    dummy_sequence = np.random.uniform(0.1, 0.9, size=(40, 195)).astype(np.float32)
    features = extract_phonological_features(dummy_sequence)

    assert features.shape == (40, PHONOLOGICAL_FEATURE_DIM)
    assert np.all(np.isfinite(features)), "Les descripteurs phonologiques ne doivent pas contenir de NaN ou d'inf."


def test_inter_hand_contact_distance():
    """Vérifie que deux index superposés donnent une distance de contact proche de 0."""
    sequence = np.zeros((10, 65, 3), dtype=np.float32)
    # Même position pour index gauche (31) et droit (52)
    sequence[:, 31, :] = [0.5, 0.5, 0.0]
    sequence[:, 52, :] = [0.5, 0.5, 0.0]

    features = extract_phonological_features(sequence)
    # L'index 4 de la sortie correspond à index_dist
    contact_dist = features[:, 4]
    np.testing.assert_allclose(contact_dist, 0.0, atol=1e-5)


def test_assemble_bistream_features_partitions_correctly():
    """Vérifie que l'assemblage bistream produit exactement (40, 416) sans NaN."""
    dummy_positions = np.random.uniform(0.1, 0.9, size=(40, 195)).astype(np.float32)
    bistream_sample = assemble_bistream_features(dummy_positions)

    assert bistream_sample.shape == (40, 416)
    assert np.all(np.isfinite(bistream_sample))

