"""Point d'entrée CLI pour télécharger les métadonnées et poses cibles de LSFB ISOL."""

from pathlib import Path
import argparse
import sys

from lsfb_live.dataset.download_config import DownloadConfig
from lsfb_live.dataset.downloader import run_dataset_download


def parse_cli_arguments() -> argparse.Namespace:
    """Analyse les arguments passés en ligne de commande.

    Returns:
        Namespace contenant les paramètres analysés.
    """
    parser = argparse.ArgumentParser(
        description="Télécharge les poses et métadonnées cibles de LSFB ISOL sans vidéos."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/lsfb_isol"),
        help="Dossier de destination (défaut : data/raw/lsfb_isol)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Nombre de signes les plus fréquents à télécharger (défaut : 100, 0 = tous)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits spécifiques à restreindre (ex: train test). Si omis, prend l'ensemble.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Nombre de connexions HTTP parallèles (défaut : 16)",
    )
    return parser.parse_args()


def main() -> int:
    """Exécute la commande de téléchargement avec affichage du récapitulatif.

    Returns:
        Code de retour système (0 en cas de succès, 1 si échec critique).
    """
    args = parse_cli_arguments()
    top_k_value = None if args.top_k <= 0 else args.top_k
    splits_tuple = tuple(args.splits) if args.splits else None

    config = DownloadConfig(
        destination_dir=args.destination,
        top_k_signs=top_k_value,
        splits=splits_tuple,
        max_parallel_connections=args.workers,
    )

    print("=== Démarrage du téléchargement LSFB ISOL ===")
    print(f"Destination : {config.destination_dir.resolve()}")
    print(f"Filtre signes : top-{config.top_k_signs if config.top_k_signs else 'tous'}")
    print(f"Repères : {', '.join(config.landmarks)}")
    print(f"Concurrence : {config.max_parallel_connections} workers")

    try:
        summary = run_dataset_download(config)
        print("\n=== Téléchargement terminé avec succès ===")
        print(f"Instances ciblées : {summary['targeted_instances']}")
        print(f"Fichiers poses totaux : {summary['total_pose_files']}")
        if summary["failed_pose_files"] > 0:
            print(f"Avertissement : {summary['failed_pose_files']} fichiers ont échoué.")
        return 0
    except Exception as error:
        print(f"\nErreur critique lors du téléchargement : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

