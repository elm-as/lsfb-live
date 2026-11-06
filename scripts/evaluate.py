"""Point d'entrée CLI pour évaluer un modèle et détailler les performances par signe."""

from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from lsfb_live.recognition.dataset import load_split_dataset
from lsfb_live.recognition.models import (
    BiGRUClassifier,
    ConvTransformerClassifier,
    ModelArchitectureConfig,
)


def parse_cli_arguments() -> argparse.Namespace:
    """Analyse les arguments passés pour l'évaluation."""
    parser = argparse.ArgumentParser(description="Évalue un point de contrôle LSFB sur le split de test.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/bigru_best.pt"),
        help="Chemin du modèle entraîné (.pt)",
    )
    parser.add_argument(
        "--model-type",
        choices=["bigru", "transformer"],
        default="bigru",
        help="Architecture du réseau (défaut : bigru)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Dossier contenant test_features.npy et class_mapping.json",
    )
    return parser.parse_args()


@torch.no_grad()
def collect_model_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collecte l'ensemble des prédictions et vérités terrain sur le dataset."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_top5: list[np.ndarray] = []

    for x_batch, y_batch in dataloader:
        logits = model(x_batch)
        _, preds = torch.max(logits, dim=1)
        _, top5_indices = torch.topk(logits, k=5, dim=1)

        all_preds.append(preds.numpy())
        all_targets.append(y_batch.numpy())
        all_top5.append(top5_indices.numpy())

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_top5, axis=0),
    )


def compute_per_class_recalls(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_to_sign: list[str],
) -> list[tuple[str, float, int]]:
    """Calcule le taux de reconnaissance (rappel) pour chaque signe individuel."""
    results: list[tuple[str, float, int]] = []
    for class_idx, sign_name in enumerate(class_to_sign):
        mask = targets == class_idx
        total = int(np.sum(mask))
        if total == 0:
            continue
        correct = int(np.sum(predictions[mask] == class_idx))
        recall = correct / total
        results.append((sign_name, recall, total))

    return sorted(results, key=lambda item: item[1], reverse=True)


def main() -> int:
    """Exécute l'évaluation exhaustive et affiche le bilan des performances."""
    args = parse_cli_arguments()
    mapping_path = args.data_dir / "class_mapping.json"
    if not mapping_path.is_file():
        print(f"Fichier de correspondance manquant : {mapping_path}", file=sys.stderr)
        return 1

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
        class_to_sign = mapping["class_to_sign"]

    test_dataset = load_split_dataset(args.data_dir, "test", augment=False)
    dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    config = ModelArchitectureConfig(num_classes=len(class_to_sign))
    model = BiGRUClassifier(config) if args.model_type == "bigru" else ConvTransformerClassifier(config)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))

    print(f"=== Évaluation de {args.model_path.name} sur {len(test_dataset)} exemples test ===")
    predictions, targets, top5_preds = collect_model_predictions(model, dataloader)

    top1_acc = np.mean(predictions == targets)
    top5_matches = np.any(top5_preds == targets[:, np.newaxis], axis=1)
    top5_acc = np.mean(top5_matches)

    class_recalls = compute_per_class_recalls(predictions, targets, class_to_sign)
    balanced_acc = np.mean([item[1] for item in class_recalls])

    print(f"\n- Top-1 Accuracy : {top1_acc * 100:5.2f}%")
    print(f"- Top-5 Accuracy : {top5_acc * 100:5.2f}%")
    print(f"- Balanced Accuracy : {balanced_acc * 100:5.2f}%")

    print("\n--- 5 signes les mieux reconnus ---")
    for sign, rec, total in class_recalls[:5]:
        print(f"  {sign:<16} : {rec * 100:5.1f}% ({total} exemples)")

    print("\n--- 5 signes les plus difficiles ---")
    for sign, rec, total in class_recalls[-5:]:
        print(f"  {sign:<16} : {rec * 100:5.1f}% ({total} exemples)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

