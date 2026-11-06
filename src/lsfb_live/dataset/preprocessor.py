"""Compilation et prétraitement des poses en tenseurs consolidés normalisés."""

from pathlib import Path
from typing import Optional
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from lsfb_live.landmarks import FEATURE_DIM, TARGET_SEQUENCE_LENGTH, transform_landmarks_pipeline


def build_class_index_mapping(
    ordered_signs: list[str],
) -> tuple[dict[str, int], list[str]]:
    """Construit le mappage bidirectionnel entre noms de signes et indices entiers.

    Args:
        ordered_signs: Liste ordonnée des gloses retenues.

    Returns:
        Tuple (dictionnaire glose -> index, liste ordonnée des gloses).
    """
    sign_to_class = {sign_name: index for index, sign_name in enumerate(ordered_signs)}
    return sign_to_class, list(ordered_signs)


def compute_inverse_frequency_weights(
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Calcule les poids inverses des fréquences pour équilibrer la fonction de perte.

    Args:
        labels: Tableau 1D des étiquettes de classe.
        num_classes: Nombre total de classes cibles.

    Returns:
        Tableau float32 de taille (num_classes,) contenant les poids normalisés.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total_samples = float(len(labels))
    safe_counts = np.maximum(counts, 1.0)
    weights = total_samples / (float(num_classes) * safe_counts)
    return (weights / np.mean(weights)).astype(np.float32)


def load_single_instance_features(
    instance_id: str,
    raw_dir: Path,
) -> Optional[np.ndarray]:
    """Charge les poses d'une instance et applique le pipeline invariant de normalisation.

    Args:
        instance_id: Identifiant unique de l'instance.
        raw_dir: Répertoire racine des données brutes (contenant poses/).

    Returns:
        Tableau float32 (TARGET_SEQUENCE_LENGTH, FEATURE_DIM) ou None si invalide/absent.
    """
    pose_path = raw_dir / "poses" / "pose" / f"{instance_id}.npy"
    lh_path = raw_dir / "poses" / "left_hand" / f"{instance_id}.npy"
    rh_path = raw_dir / "poses" / "right_hand" / f"{instance_id}.npy"

    if not (pose_path.is_file() and lh_path.is_file() and rh_path.is_file()):
        return None

    try:
        pose = np.load(pose_path)
        left_hand = np.load(lh_path)
        right_hand = np.load(rh_path)
    except (ValueError, OSError):
        return None

    if len(pose) == 0:
        return None

    return transform_landmarks_pipeline(pose, left_hand, right_hand)


def compile_instances_to_arrays(
    split_instances: pd.DataFrame,
    sign_to_class: dict[str, int],
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Transforme un sous-ensemble d'instances en matrices de tenseurs et étiquettes.

    Args:
        split_instances: DataFrame avec colonnes 'id' et 'sign'.
        sign_to_class: Dictionnaire de correspondance signe -> entier.
        raw_dir: Dossier des fichiers bruts.

    Returns:
        Tuple (features de forme (N, 40, 195), labels de forme (N,)).
    """
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []

    for _, row in split_instances.iterrows():
        instance_id = row["id"]
        sign_label = row["sign"]
        if sign_label not in sign_to_class:
            continue

        features = load_single_instance_features(instance_id, raw_dir)
        if features is None:
            continue

        features_list.append(features)
        labels_list.append(sign_to_class[sign_label])

    if not features_list:
        empty_features = np.zeros((0, TARGET_SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        empty_labels = np.zeros((0,), dtype=np.int64)
        return empty_features, empty_labels

    stacked_features = np.stack(features_list, axis=0).astype(np.float32)
    stacked_labels = np.array(labels_list, dtype=np.int64)
    return stacked_features, stacked_labels


def save_split_tensors(
    split_name: str,
    tensors: tuple[np.ndarray, np.ndarray],
    output_dir: Path,
) -> None:
    """Enregistre sur disque les caractéristiques et labels d'un split.

    Args:
        split_name: Nom du split (ex: 'train', 'val', 'test').
        tensors: Tuple (features, labels).
        output_dir: Répertoire cible d'écriture.
    """
    features, labels = tensors
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{split_name}_features.npy", features)
    np.save(output_dir / f"{split_name}_labels.npy", labels)


def compile_top_k_dataset(
    raw_dir: Path,
    output_dir: Path,
    top_k: int = 100,
) -> dict[str, int]:
    """Compile l'ensemble du dataset top_k en tenseurs d'entraînement, validation et test.

    Args:
        raw_dir: Dossier des poses brutes LSFB ISOL.
        output_dir: Dossier destination des tenseurs consolidés.
        top_k: Nombre de signes cibles (100 par défaut).

    Returns:
        Dictionnaire récapitulant la taille de chaque split.
    """
    instances_df = pd.read_csv(raw_dir / "instances.csv")
    occurrences_df = pd.read_csv(raw_dir / "metadata" / "sign_occurrences.csv")

    top_signs = list(occurrences_df.head(top_k)["sign"])
    sign_to_class, class_to_sign = build_class_index_mapping(top_signs)

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_payload = {
        "sign_to_class": sign_to_class,
        "class_to_sign": class_to_sign,
    }
    with open(output_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=2, ensure_ascii=False)

    df_top = instances_df[instances_df["sign"].isin(set(top_signs))].copy()

    def get_fold_ids(fold_indices: list[int]) -> set[str]:
        ids: set[str] = set()
        for idx in fold_indices:
            with open(raw_dir / "metadata" / "splits" / f"fold_{idx}.json", "r", encoding="utf-8") as file_p:
                ids.update(json.load(file_p))
        return ids

    split_definitions = {
        "train": get_fold_ids([2, 3]),
        "val": get_fold_ids([4]),
        "test": get_fold_ids([0, 1]),
    }

    summary_counts: dict[str, int] = {}

    for split_name, instance_ids in split_definitions.items():
        print(f"Compilation du split '{split_name}'...")
        split_df = df_top[df_top["id"].isin(instance_ids)]
        tensors = compile_instances_to_arrays(split_df, sign_to_class, raw_dir)
        save_split_tensors(split_name, tensors, output_dir)
        summary_counts[split_name] = len(tensors[1])

        if split_name == "train":
            class_weights = compute_inverse_frequency_weights(tensors[1], num_classes=top_k)
            np.save(output_dir / "class_weights.npy", class_weights)

    return summary_counts
