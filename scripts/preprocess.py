"""Point d'entrée CLI pour compiler les poses brutes en tenseurs normalisés d'entraînement."""

from pathlib import Path
import argparse
import sys

from lsfb_live.dataset.preprocessor import compile_top_k_dataset


def parse_cli_arguments() -> argparse.Namespace:
    """Analyse les arguments CLI pour le prétraitement.

    Returns:
        Namespace des arguments analysés.
    """
    parser = argparse.ArgumentParser(
        description="Compile et normalise les poses LSFB en tenseurs d'entraînement consolidés."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/lsfb_isol"),
        help="Dossier contenant les poses brutes (défaut : data/raw/lsfb_isol)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Dossier de sortie pour les tenseurs consolidés (défaut : data/processed)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Nombre de signes à prétraiter (défaut : 100)",
    )
    return parser.parse_args()


def main() -> int:
    """Exécute la compilation des données et affiche le bilan volumétrique.

    Returns:
        0 en cas de succès, 1 si erreur.
    """
    args = parse_cli_arguments()
    print("=== Démarrage du prétraitement des tenseurs LSFB ===")
    print(f"Source brute : {args.raw_dir.resolve()}")
    print(f"Destination : {args.output_dir.resolve()}")
    print(f"Top-K : {args.top_k} signes")

    try:
        summary = compile_top_k_dataset(args.raw_dir, args.output_dir, top_k=args.top_k)
        print("\n=== Prétraitement terminé avec succès ===")
        for split_name, count in summary.items():
            print(f"- Split {split_name} : {count} instances compilées")
        return 0
    except Exception as error:
        print(f"\nErreur critique lors du prétraitement : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

