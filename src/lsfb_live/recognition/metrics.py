"""Calcul des métriques d'évaluation adaptées au déséquilibre des classes."""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class EvaluationMetrics:
    """Conteneur des métriques quantitatives d'évaluation d'un modèle.

    Attributes:
        loss: Perte moyenne sur le jeu de test.
        top1_accuracy: Précision Top-1 standard (0.0 à 1.0).
        top5_accuracy: Précision Top-5 (0.0 à 1.0).
        balanced_accuracy: Précision moyenne par classe (macro-recall).
    """

    loss: float
    top1_accuracy: float
    top5_accuracy: float
    balanced_accuracy: float


def compute_top_k_accuracies(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Calcule la précision Top-1 et Top-5 à partir des logits et cibles.

    Args:
        logits: Logits bruts du modèle (N, C).
        targets: Classes cibles (N,).

    Returns:
        Tuple (top1_accuracy, top5_accuracy).
    """
    total = len(targets)
    if total == 0:
        return 0.0, 0.0

    _, top1_preds = torch.max(logits, dim=1)
    top1_correct = (top1_preds == targets).sum().item()

    k = min(5, logits.shape[1])
    _, top5_preds = torch.topk(logits, k=k, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top5_preds)
    top5_correct = (top5_preds == targets_expanded).any(dim=1).sum().item()

    return float(top1_correct / total), float(top5_correct / total)


def compute_balanced_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> float:
    """Calcule la précision équilibrée (moyenne des rappels par classe).

    Args:
        predictions: Prédictions 1D du modèle.
        targets: Vérités terrain 1D.
        num_classes: Nombre total de classes.

    Returns:
        Balanced accuracy entre 0.0 et 1.0.
    """
    recalls: list[float] = []
    for class_idx in range(num_classes):
        mask = targets == class_idx
        class_total = int(np.sum(mask))
        if class_total == 0:
            continue
        class_correct = int(np.sum(predictions[mask] == class_idx))
        recalls.append(class_correct / class_total)

    if not recalls:
        return 0.0
    return float(np.mean(recalls))

