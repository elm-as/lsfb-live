"""Indexation et filtrage des instances du dataset LSFB ISOL."""

from pathlib import Path
import json
import pandas as pd


def load_sign_occurrences(metadata_dir: Path) -> pd.DataFrame:
    """Charge la table de fréquence des signes du dataset.

    Args:
        metadata_dir: Répertoire contenant les fichiers de métadonnées.

    Returns:
        DataFrame contenant les colonnes 'sign' et 'occurrences', trié par fréquence.

    Raises:
        FileNotFoundError: Si le fichier sign_occurrences.csv est absent.
    """
    occurrences_path = metadata_dir / "metadata" / "sign_occurrences.csv"
    if not occurrences_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {occurrences_path}")
    return pd.read_csv(occurrences_path)


def load_instances_metadata(root_dir: Path) -> pd.DataFrame:
    """Charge le catalogue complet de toutes les instances segmentées.

    Args:
        root_dir: Répertoire racine du dataset LSFB ISOL (contenant instances.csv).

    Returns:
        DataFrame avec colonnes: id, sign, signer, start, end.

    Raises:
        FileNotFoundError: Si instances.csv n'existe pas.
    """
    instances_path = root_dir / "instances.csv"
    if not instances_path.is_file():
        raise FileNotFoundError(f"Catalogue d'instances manquant : {instances_path}")
    return pd.read_csv(instances_path)


def load_split_instance_ids(root_dir: Path, split_name: str) -> set[str]:
    """Récupère l'ensemble des identifiants d'instances appartenant à un split.

    Args:
        root_dir: Répertoire racine du dataset.
        split_name: Nom du split (ex: 'train', 'test', 'fold_0').

    Returns:
        Set des identifiants d'instances.

    Raises:
        FileNotFoundError: Si le split demandé n'existe pas.
    """
    split_path = root_dir / "metadata" / "splits" / f"{split_name}.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Fichier de split introuvable : {split_path}")

    with open(split_path, "r", encoding="utf-8") as file_pointer:
        return set(json.load(file_pointer))


def select_top_signs_instances(
    instances_df: pd.DataFrame,
    occurrences_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Filtre les instances pour ne retenir que les top_k signes les plus fréquents.

    Args:
        instances_df: DataFrame des instances segmentées.
        occurrences_df: DataFrame des fréquences par signe.
        top_k: Nombre de signes à conserver.

    Returns:
        Sous-ensemble filtré d'instances_df.
    """
    if top_k <= 0:
        return instances_df.iloc[0:0].copy()

    retained_signs = set(occurrences_df.head(top_k)["sign"])
    return instances_df[instances_df["sign"].isin(retained_signs)].copy()

