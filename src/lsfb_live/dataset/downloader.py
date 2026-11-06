"""Moteur de téléchargement résilient pour le dataset LSFB ISOL."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urljoin
import os
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

from lsfb_live.dataset.dataset_index import (
    load_instances_metadata,
    load_sign_occurrences,
    load_split_instance_ids,
    select_top_signs_instances,
)
from lsfb_live.dataset.download_config import DownloadConfig


def create_resilient_session(config: DownloadConfig) -> requests.Session:
    """Initialise une session HTTP avec reprise automatique sur pannes transitoires.

    Args:
        config: Paramètres de téléchargement spécifiant timeouts et retries.

    Returns:
        Session requests configurée avec adaptateur et backoff.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=config.max_retries,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=config.max_parallel_connections,
        pool_maxsize=config.max_parallel_connections * 2,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def build_metadata_relative_paths() -> list[str]:
    """Construit la liste des chemins relatifs de tous les fichiers de métadonnées.

    Returns:
        Liste des chemins relatifs normalisés.
    """
    metadata_paths = [
        "instances.csv",
        "instances_special.csv",
        "metadata/sign_occurrences.csv",
        "metadata/sign_to_index.csv",
    ]
    split_names = ["all", "mini_sample", "test", "train"] + [f"fold_{k}" for k in range(5)]
    for split_name in split_names:
        metadata_paths.append(f"metadata/splits/{split_name}.json")
    return metadata_paths


def download_single_file(
    relative_path: str,
    config: DownloadConfig,
    session: requests.Session,
) -> bool:
    """Télécharge un fichier distant si absent localement ou vide.

    Args:
        relative_path: Chemin relatif de la ressource.
        config: Configuration du téléchargement.
        session: Session HTTP partagée.

    Returns:
        True si le fichier est présent ou téléchargé avec succès, False sinon.
    """
    local_target = config.destination_dir / relative_path
    if local_target.is_file() and local_target.stat().st_size > 0:
        return True

    url = urljoin(f"{config.base_url}/", quote(relative_path))
    try:
        response = session.get(url, timeout=config.timeout_seconds)
        if response.status_code != 200 or response.headers.get("Content-Type") == "text/html":
            return False

        local_target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = local_target.with_suffix(f"{local_target.suffix}.part")
        temp_target.write_bytes(response.content)
        temp_target.replace(local_target)
        return True
    except (requests.RequestException, OSError):
        return False


def download_relative_paths_batch(
    relative_paths: list[str],
    config: DownloadConfig,
    description: str,
) -> list[str]:
    """Télécharge un lot de fichiers en parallèle et rapporte les échecs.

    Args:
        relative_paths: Liste des chemins relatifs à rapatrier.
        config: Configuration contenant la concurrence et le dossier cible.
        description: Intitulé affiché dans la barre de progression.

    Returns:
        Liste des chemins n'ayant pas pu être téléchargés.
    """
    if not relative_paths:
        return []

    failures: list[str] = []
    session = create_resilient_session(config)

    with ThreadPoolExecutor(max_workers=config.max_parallel_connections) as executor:
        future_map = {
            executor.submit(download_single_file, rel_path, config, session): rel_path
            for rel_path in relative_paths
        }
        progress_bar = tqdm(total=len(relative_paths), desc=description, unit="fichiers")
        for future in as_completed(future_map):
            relative_path = future_map[future]
            try:
                success = future.result()
                if not success:
                    failures.append(relative_path)
            except Exception:
                failures.append(relative_path)
            finally:
                progress_bar.update(1)
        progress_bar.close()

    return failures


def collect_target_pose_paths(
    instance_ids: set[str],
    landmarks: tuple[str, ...],
) -> list[str]:
    """Produit la liste exhaustive des chemins de poses à télécharger.

    Args:
        instance_ids: Ensemble des identifiants d'instances sélectionnés.
        landmarks: Types de squelettes à extraire ('pose', 'left_hand', etc.).

    Returns:
        Liste ordonnée de tous les chemins relatifs correspondants.
    """
    pose_paths: list[str] = []
    for instance_id in sorted(instance_ids):
        for landmark in landmarks:
            pose_paths.append(f"poses/{landmark}/{instance_id}.npy")
    return pose_paths


def resolve_targeted_instance_ids(config: DownloadConfig) -> set[str]:
    """Détermine les identifiants d'instances à télécharger selon la configuration.

    Args:
        config: Configuration portant sur les splits et le top_k_signs.

    Returns:
        Ensemble des identifiants d'instances retenus.
    """
    instances_df = load_instances_metadata(config.destination_dir)

    if config.top_k_signs is not None:
        occurrences_df = load_sign_occurrences(config.destination_dir)
        instances_df = select_top_signs_instances(instances_df, occurrences_df, config.top_k_signs)

    targeted_ids = set(instances_df["id"])

    if config.splits is not None:
        allowed_split_ids: set[str] = set()
        for split_name in config.splits:
            allowed_split_ids.update(load_split_instance_ids(config.destination_dir, split_name))
        targeted_ids.intersection_update(allowed_split_ids)

    return targeted_ids


def run_dataset_download(config: DownloadConfig) -> dict[str, int]:
    """Exécute l'orchestration complète du téléchargement de LSFB ISOL.

    Args:
        config: Configuration globale du téléchargement.

    Returns:
        Dictionnaire récapitulant le nombre de fichiers traités et échoués.
    """
    config.destination_dir.mkdir(parents=True, exist_ok=True)

    metadata_paths = build_metadata_relative_paths()
    metadata_failures = download_relative_paths_batch(
        metadata_paths, config, description="Métadonnées LSFB"
    )
    if metadata_failures:
        raise RuntimeError(
            f"Échec de téléchargement des métadonnées critiques : {len(metadata_failures)} fichiers."
        )

    targeted_instance_ids = resolve_targeted_instance_ids(config)
    pose_paths = collect_target_pose_paths(targeted_instance_ids, config.landmarks)

    pose_failures = download_relative_paths_batch(
        pose_paths, config, description=f"Poses ({len(targeted_instance_ids)} instances)"
    )

    return {
        "targeted_instances": len(targeted_instance_ids),
        "total_pose_files": len(pose_paths),
        "failed_pose_files": len(pose_failures),
    }

