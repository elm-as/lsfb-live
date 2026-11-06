"""Point d'entrée CLI pour entraîner les modèles de reconnaissance LSFB."""

from pathlib import Path
import argparse
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from lsfb_live.recognition.dataset import load_enriched_split_dataset, load_split_dataset
from lsfb_live.recognition.models import (
    BiGRUClassifier,
    ConvTransformerClassifier,
    ModelArchitectureConfig,
    MSSTClassifier,
)
from lsfb_live.recognition.trainer import (
    TrainingHyperparameters,
    evaluate_dataset,
    train_and_select_best_model,
)


def parse_cli_arguments() -> argparse.Namespace:
    """Analyse les arguments d'entraînement passés en CLI."""
    parser = argparse.ArgumentParser(description="Entraîne un modèle de reconnaissance LSFB (BiGRU, Transformer, MSST).")
    parser.add_argument(
        "--model",
        choices=["bigru", "transformer", "msst"],
        default="msst",
        help="Type d'architecture à entraîner (défaut : msst)",
    )
    parser.add_argument(
        "--use-velocity",
        action="store_true",
        default=True,
        help="Enrichit les tenseurs avec les vitesses instantanées (390 features, défaut: True)",
    )
    parser.add_argument(
        "--no-velocity",
        action="store_false",
        dest="use_velocity",
        help="Désactive l'enrichissement par vitesse (195 features)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Dossier contenant les tenseurs préprocessés",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=None,
        help="Chemin de sauvegarde des poids optimaux (.pt)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Nombre maximal d'époques (défaut : 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille des batchs (défaut : 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taux d'apprentissage (défaut : 1e-3)")
    return parser.parse_args()


def build_model_instance(model_name: str, config: ModelArchitectureConfig) -> torch.nn.Module:
    """Instancie le modèle demandé selon la configuration."""
    if model_name == "bigru":
        return BiGRUClassifier(config)
    if model_name == "transformer":
        return ConvTransformerClassifier(config)
    if model_name == "msst":
        return MSSTClassifier(config)
    raise ValueError(f"Modèle non reconnu : {model_name}")


def main() -> int:
    """Orchestre le chargement, l'entraînement et l'évaluation finale sur le test officiel."""
    args = parse_cli_arguments()
    data_dir: Path = args.data_dir
    checkpoint_path = args.output_model or Path(f"models/{args.model}_best.pt")

    print(f"=== Entraînement du modèle '{args.model.upper()}' ===")
    print(f"Données : {data_dir.resolve()} | Vitesses : {args.use_velocity}")
    print(f"Époques : {args.epochs} | Batch size : {args.batch_size} | LR : {args.lr}")

    dataset_loader = load_enriched_split_dataset if args.use_velocity else load_split_dataset
    train_dataset = dataset_loader(data_dir, "train", augment=True)
    val_dataset = dataset_loader(data_dir, "val", augment=False)
    test_dataset = dataset_loader(data_dir, "test", augment=False)

    print(f"Tailles : Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    class_weights_path = data_dir / "class_weights.npy"
    class_weights = np.load(class_weights_path) if class_weights_path.is_file() else None

    feature_dimension = 390 if args.use_velocity else 195
    arch_config = ModelArchitectureConfig(input_dim=feature_dimension, num_classes=100)
    model = build_model_instance(args.model, arch_config)
    hparams = TrainingHyperparameters(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    result = train_and_select_best_model(
        model=model,
        datasets=(train_dataset, val_dataset),
        hparams=hparams,
        class_weights=class_weights,
        checkpoint_path=checkpoint_path,
    )

    print(f"\nMeilleur score validation (Balanced Acc) : {result['best_val_score'] * 100:.2f}%")
    print(f"Poids sauvegardés dans : {checkpoint_path.resolve()}")

    print("\n=== Évaluation sur le test officiel (40 signeurs non vus) ===")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = evaluate_dataset(model, test_loader, criterion)

    print(f"Test Loss : {test_metrics.loss:.4f}")
    print(f"Test Top-1 Accuracy : {test_metrics.top1_accuracy * 100:.2f}%")
    print(f"Test Top-5 Accuracy : {test_metrics.top5_accuracy * 100:.2f}%")
    print(f"Test Balanced Accuracy : {test_metrics.balanced_accuracy * 100:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
