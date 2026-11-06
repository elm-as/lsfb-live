"""Configuration typée pour le téléchargement du dataset LSFB."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DownloadConfig:
    """Paramètres immuables régissant le téléchargement des poses et métadonnées.

    Attributes:
        destination_dir: Répertoire racine de stockage des données brutes.
        dataset_name: Identifiant du dataset ('isol' par défaut).
        base_url: URL racine du serveur UNamur distribuant le dataset.
        landmarks: Liste des repères à télécharger ('pose', 'left_hand', 'right_hand').
        top_k_signs: Nombre de signes les plus fréquents à cibler (None = tous).
        splits: Splits spécifiques à filtrer (None = 'all').
        max_parallel_connections: Nombre de workers simultanés pour le pool HTTP.
        timeout_seconds: Délai d'attente maximum par requête HTTP en secondes.
        max_retries: Nombre de tentatives automatiques sur erreur transitoire.
    """

    destination_dir: Path
    dataset_name: str = "isol"
    base_url: str = "https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol"
    landmarks: tuple[str, ...] = ("pose", "left_hand", "right_hand")
    top_k_signs: Optional[int] = 100
    splits: Optional[tuple[str, ...]] = None
    max_parallel_connections: int = 12
    timeout_seconds: int = 30
    max_retries: int = 4

