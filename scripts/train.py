"""Point d'entrée CLI pour entraîner les modèles de reconnaissance LSFB."""

import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from lsfb_live.recognition.dataset import (
    load_enriched_split_dataset,
    load_phonological_split_dataset,
    load_split_dataset,
)
from lsfb_live.recognition.models import (
    BiGRUClassifier,
    BiStreamMSSTClassifier,
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
    parser = argparse.ArgumentParser(description="Entraîne un modèle de reconnaissance LSFB (BiGRU, Transformer, MSST, BiStream).")
    parser.add_argument(
        "--model",
        choices=["bigru", "transformer", "msst", "bistream"],
        default="bistream",
        help="Type d'architecture à entraîner (défaut : bistream)",
    )
    parser.add_argument(
        "--use-velocity",
        action="store_true",
        default=True,
        help="Enrichit les tenseurs avec les vitesses instantanées (390 features pour MSST, défaut: True)",
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
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Nombre de classes cibles (déduit automatiquement si omis)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Nombre maximal d'époques (défaut : 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille des batchs (défaut : 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taux d'apprentissage (défaut : 1e-3)")
    parser.add_argument("--focal", action="store_true", default=True, help="Active la Focal Cross-Entropy (défaut: True)")
    parser.add_argument("--no-focal", action="store_false", dest="focal", help="Désactive la Focal Loss")
    return parser.parse_args()


def build_model_instance(model_name: str, config: ModelArchitectureConfig) -> torch.nn.Module:
    """Instancie le modèle demandé selon la configuration."""
    if model_name == "bigru":
        return BiGRUClassifier(config)
    if model_name == "transformer":
        return ConvTransformerClassifier(config)
    if model_name == "msst":
        return MSSTClassifier(config)
    if model_name == "bistream":
        return BiStreamMSSTClassifier(config)
    raise ValueError(f"Modèle non reconnu : {model_name}")


def main() -> int:
    """Orchestre le chargement, l'entraînement et l'évaluation finale sur le test officiel."""
    args = parse_cli_arguments()
    data_dir: Path = args.data_dir
    checkpoint_path = args.output_model or Path(f"models/{args.model}_best.pt")

    print(f"=== Entraînement du modèle '{args.model.upper()}' ===")
    print(f"Données : {data_dir.resolve()} | Vitesses : {args.use_velocity}")
    print(f"Époques : {args.epochs} | Batch size : {args.batch_size} | LR : {args.lr}")

    is_bistream = args.model == "bistream"
    if is_bistream:
        dataset_loader = load_phonological_split_dataset
        feature_dimension = 416
        hidden_dim = 192
    else:
        dataset_loader = load_enriched_split_dataset if args.use_velocity else load_split_dataset
        feature_dimension = 390 if args.use_velocity else 195
        hidden_dim = 128

    train_dataset = dataset_loader(data_dir, "train", augment=True)
    val_dataset = dataset_loader(data_dir, "val", augment=False)
    test_dataset = dataset_loader(data_dir, "test", augment=False)

    print(f"Tailles : Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    class_weights_path = data_dir / "class_weights.npy"
    class_weights = np.load(class_weights_path) if class_weights_path.is_file() else None

    mapping_path = data_dir / "class_mapping.json"
    if args.num_classes is not None:
        target_num_classes = args.num_classes
    elif mapping_path.is_file():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_dict = json.load(f)
            target_num_classes = len(mapping_dict.get("class_to_sign", []))
    elif class_weights is not None:
        target_num_classes = len(class_weights)
    else:
        target_num_classes = 100

    print(f"Nombre de classes actives : {target_num_classes} | Dim features : {feature_dimension}")
    arch_config = ModelArchitectureConfig(
        input_dim=feature_dimension,
        hidden_dim=hidden_dim,
        num_classes=target_num_classes,
    )
    model = build_model_instance(args.model, arch_config)
    hparams = TrainingHyperparameters(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_focal=args.focal if is_bistream else False,
    )

    result = train_and_select_best_model(
        model=model,
        datasets=(train_dataset, val_dataset),
        hparams=hparams,
        class_weights=class_weights,
        checkpoint_path=checkpoint_path,
    )

    if mapping_path.is_file():
        dest_mapping = checkpoint_path.parent / "class_mapping.json"
        if dest_mapping != mapping_path:
            shutil.copyfile(mapping_path, dest_mapping)
            print(f"Mappage des classes copié vers : {dest_mapping.resolve()}")

    print(f"\nMeilleur score validation (Balanced Acc) : {result['best_val_score'] * 100:.2f}%", flush=True)
    print(f"Poids sauvegardés dans : {checkpoint_path.resolve()}", flush=True)

    print("\n=== Évaluation sur le test officiel (40 signeurs non vus) ===", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = evaluate_dataset(model, test_loader, criterion)

    print(f"Test Loss : {test_metrics.loss:.4f}", flush=True)
    print(f"Test Top-1 Accuracy : {test_metrics.top1_accuracy * 100:.2f}%", flush=True)
    print(f"Test Top-5 Accuracy : {test_metrics.top5_accuracy * 100:.2f}%", flush=True)
    print(f"Test Balanced Accuracy : {test_metrics.balanced_accuracy * 100:.2f}%", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
