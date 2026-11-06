"""Tests unitaires pour la compilation et le calcul des poids du prétraitement."""

import numpy as np
import pandas as pd

from lsfb_live.dataset.preprocessor import (
    build_class_index_mapping,
    compute_inverse_frequency_weights,
)


def test_build_class_index_mapping_is_bijective():
    """Vérifie la bijection entre indices entiers et gloses textuelles."""
    signs = ["AUSSI", "BONJOUR", "MERCI"]
    sign_to_class, class_to_sign = build_class_index_mapping(signs)

    assert len(sign_to_class) == 3
    assert len(class_to_sign) == 3
    for idx, sign in enumerate(signs):
        assert sign_to_class[sign] == idx
        assert class_to_sign[idx] == sign


def test_compute_inverse_frequency_weights_penalizes_frequent_classes():
    """Vérifie que la classe minoritaire reçoit un poids strictement supérieur à la majoritaire."""
    # Classe 0 présente 100 fois, Classe 1 présente 10 fois
    labels = np.array([0] * 100 + [1] * 10)
    weights = compute_inverse_frequency_weights(labels, num_classes=2)

    assert len(weights) == 2
    assert weights[1] > weights[0]
    assert np.isclose(np.mean(weights), 1.0, atol=1e-4)

