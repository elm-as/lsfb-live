"""Boucle d'entraînement PyTorch optimisée pour CPU avec arrêt anticipé."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lsfb_live.recognition.dataset import SignPoseDataset
from lsfb_live.recognition.metrics import (
    EvaluationMetrics,
    compute_balanced_accuracy,
    compute_top_k_accuracies,
)


@dataclass(frozen=True)
class TrainingHyperparameters:
    """Hyperparamètres d'entraînement du modèle.

    Attributes:
        epochs: Nombre maximum d'époques (25).
        batch_size: Taille des mini-lots (64).
        learning_rate: Taux d'apprentissage initial (1e-3).
        weight_decay: Régularisation L2 (1e-4).
        label_smoothing: Lissage d'étiquette pour limiter la sur-confiance (0.05).
        patience: Époques sans amélioration avant arrêt anticipé (6).
        device: Dispositif d'exécution ('cpu').
    """

    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 6
    device: str = "cpu"


def train_single_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Exécute une époque d'entraînement et retourne la perte moyenne."""
    model.train()
    total_loss = 0.0
    batch_count = 0

    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / max(batch_count, 1)


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    num_classes: int = 100,
) -> EvaluationMetrics:
    """Évalue le modèle sur un jeu de données complet et compile les métriques."""
    model.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for x_batch, y_batch in dataloader:
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        all_logits.append(logits)
        all_targets.append(y_batch)

    stacked_logits = torch.cat(all_logits, dim=0)
    stacked_targets = torch.cat(all_targets, dim=0)

    avg_loss = total_loss / max(len(stacked_targets), 1)
    top1, top5 = compute_top_k_accuracies(stacked_logits, stacked_targets)

    predictions = torch.argmax(stacked_logits, dim=1).numpy()
    targets_np = stacked_targets.numpy()
    balanced_acc = compute_balanced_accuracy(predictions, targets_np, num_classes)

    return EvaluationMetrics(
        loss=avg_loss,
        top1_accuracy=top1,
        top5_accuracy=top5,
        balanced_accuracy=balanced_acc,
    )


def train_and_select_best_model(
    model: nn.Module,
    datasets: tuple[SignPoseDataset, SignPoseDataset],
    hparams: TrainingHyperparameters,
    class_weights: Optional[np.ndarray] = None,
    checkpoint_path: Optional[Path] = None,
) -> dict[str, object]:
    """Orchestre l'entraînement complet avec ordonnanceur et sauvegarde du meilleur point de contrôle."""
    train_dataset, val_dataset = datasets
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False)

    device = torch.device(hparams.device)
    model.to(device)

    weights_tensor = torch.from_numpy(class_weights).float().to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=hparams.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams.epochs)

    best_val_score = -1.0
    best_metrics: Optional[EvaluationMetrics] = None
    epochs_without_improvement = 0

    for epoch in range(1, hparams.epochs + 1):
        train_loss = train_single_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate_dataset(model, val_loader, criterion)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{hparams.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics.loss:.4f} | "
            f"Val Top-1: {val_metrics.top1_accuracy * 100:.2f}% | "
            f"Val Top-5: {val_metrics.top5_accuracy * 100:.2f}% | "
            f"Val Balanced: {val_metrics.balanced_accuracy * 100:.2f}%"
        )

        current_score = val_metrics.balanced_accuracy
        if current_score > best_val_score:
            best_val_score = current_score
            best_metrics = val_metrics
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hparams.patience:
                print(f"Arrêt anticipé à l'époque {epoch} (pas de gain depuis {hparams.patience} époques).")
                break

    return {
        "best_val_score": best_val_score,
        "best_metrics": best_metrics,
    }

