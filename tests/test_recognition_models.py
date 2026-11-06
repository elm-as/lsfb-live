"""Tests unitaires pour les architectures de modèles et calculs de métriques."""

import numpy as np
import torch

from lsfb_live.recognition.metrics import (
    compute_balanced_accuracy,
    compute_top_k_accuracies,
)
from lsfb_live.recognition.models import (
    BiGRUClassifier,
    ConvTransformerClassifier,
    ModelArchitectureConfig,
)


def test_bigru_classifier_produces_expected_output_shape():
    """Vérifie la forme des logits (Batch, NumClasses) du BiGRU."""
    config = ModelArchitectureConfig(input_dim=195, hidden_dim=64, num_classes=10)
    model = BiGRUClassifier(config)

    dummy_input = torch.randn(4, 40, 195)
    logits = model(dummy_input)

    assert logits.shape == (4, 10)


def test_conv_transformer_classifier_produces_expected_output_shape():
    """Vérifie la forme des logits (Batch, NumClasses) du Transformer."""
    config = ModelArchitectureConfig(input_dim=195, hidden_dim=64, num_classes=10)
    model = ConvTransformerClassifier(config)

    dummy_input = torch.randn(4, 40, 195)
    logits = model(dummy_input)

    assert logits.shape == (4, 10)


def test_compute_top_k_accuracies_calculates_correctly():
    """Vérifie les ratios Top-1 et Top-5 sur des prédictions maîtrisées."""
    # 2 exemples : le 1er exact, le 2nd dans le top-2
    logits = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 5.0, 10.0],
    ])
    targets = torch.tensor([0, 1])

    top1, top5 = compute_top_k_accuracies(logits, targets)
    assert top1 == 0.5
    assert top5 == 1.0


def test_compute_balanced_accuracy_averages_per_class():
    """Vérifie le calcul de balanced accuracy en présence de classes déséquilibrées."""
    predictions = np.array([0, 0, 1, 1])
    targets = np.array([0, 0, 1, 0])
    # Classe 0 : 3 cibles, 2 corrects -> rappel = 2/3
    # Classe 1 : 1 cible, 1 correct -> rappel = 1/1
    # Moyenne = (2/3 + 1)/2 = 5/6 ~= 0.8333
    acc = compute_balanced_accuracy(predictions, targets, num_classes=2)
    assert np.isclose(acc, 5.0 / 6.0, atol=1e-4)

