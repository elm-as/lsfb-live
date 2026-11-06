"""Dataset PyTorch pour les tenseurs de séquences de squelettes LSFB."""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset


# Indices des paires symétriques de repères du corps
_SYMMETRIC_POSE_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),  # Yeux, oreilles, bouche
    (11, 12), (13, 14), (15, 16),             # Épaules, coudes, poignets
    (17, 18), (19, 20), (21, 22),             # Doigts supérieurs de pose
]

# Construction statique de la table de permutation pour 65 repères
_LANDMARK_SWAP_MAP = list(range(65))
for _p1, _p2 in _SYMMETRIC_POSE_PAIRS:
    _LANDMARK_SWAP_MAP[_p1] = _p2
    _LANDMARK_SWAP_MAP[_p2] = _p1
for _i in range(21):
    _lh = 23 + _i
    _rh = 44 + _i
    _LANDMARK_SWAP_MAP[_lh] = _rh
    _LANDMARK_SWAP_MAP[_rh] = _lh


def apply_handedness_mirror(sequence: np.ndarray) -> np.ndarray:
    """Inversion miroir horizontale et permutation coordonnée gauche/droite.

    Garantit l'invariance aux signeurs gauchers et droitiers.

    Args:
        sequence: Tableau float32 de forme (T, 195) ou (T, 390).

    Returns:
        Tableau miroir de même forme.
    """
    total_features = sequence.shape[-1]
    is_enriched = total_features == 390
    channels = [sequence[:, :195], sequence[:, 195:]] if is_enriched else [sequence]

    mirrored_channels: list[np.ndarray] = []
    for chan in channels:
        reshaped = chan.reshape(-1, 65, 3).copy()
        reshaped[:, :, 0] = -reshaped[:, :, 0]  # Inversion de l'axe X
        permuted = reshaped[:, _LANDMARK_SWAP_MAP, :]
        mirrored_channels.append(permuted.reshape(-1, 195))

    if is_enriched:
        return np.concatenate(mirrored_channels, axis=-1).astype(np.float32)
    return mirrored_channels[0].astype(np.float32)


def apply_training_augmentation(
    sequence: np.ndarray,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Applique des augmentations (symétrie miroir, bruit gaussien, jitter d'échelle).

    Args:
        sequence: Tableau float32 de forme (40, 195) ou (40, 390).
        noise_std: Écart-type du bruit gaussien ajouté.

    Returns:
        Nouvelle séquence augmentée.
    """
    augmented = sequence.copy()

    # 1. Symétrie miroir avec probabilité 0.5
    if np.random.rand() > 0.5:
        augmented = apply_handedness_mirror(augmented)

    # 2. Jitter d'échelle et bruit additif
    scale_jitter = np.random.uniform(0.95, 1.05)
    noise = np.random.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
    augmented = (augmented * scale_jitter) + noise
    return augmented.astype(np.float32)


class SignPoseDataset(Dataset):
    """Dataset PyTorch gérant les tenseurs de poses et leurs étiquettes de classe.

    Args:
        features: Tableau de forme (N, 40, 195) ou (N, 40, 390).
        labels: Tableau des classes cibles (N,).
        augment: Active l'augmentation de données aléatoire pour l'entraînement.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        if len(features) != len(labels):
            raise ValueError(f"Incohérence de taille : {len(features)} features vs {len(labels)} labels")

        self._features = features
        self._labels = labels
        self._augment = augment

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = np.array(self._features[index], copy=True)
        if self._augment:
            sample = apply_training_augmentation(sample)

        tensor_x = torch.from_numpy(sample).float()
        tensor_y = torch.tensor(self._labels[index], dtype=torch.long)
        return tensor_x, tensor_y


def load_split_dataset(
    processed_dir: Path,
    split_name: str,
    augment: bool = False,
) -> SignPoseDataset:
    """Instancie un SignPoseDataset à partir des fichiers .npy enregistrés sur disque.

    Args:
        processed_dir: Répertoire contenant les fichiers préprocessés.
        split_name: Nom du split ('train', 'val', 'test').
        augment: Si True, active l'augmentation aléatoire.

    Returns:
        Instance de SignPoseDataset prête pour DataLoader.
    """
    feat_path = processed_dir / f"{split_name}_features.npy"
    lab_path = processed_dir / f"{split_name}_labels.npy"

    if not (feat_path.is_file() and lab_path.is_file()):
        raise FileNotFoundError(f"Fichiers manquants pour le split {split_name} dans {processed_dir}")

    features = np.load(feat_path, mmap_mode="r")
    labels = np.load(lab_path)
    return SignPoseDataset(features, labels, augment=augment)


def load_enriched_split_dataset(
    processed_dir: Path,
    split_name: str,
    augment: bool = False,
) -> SignPoseDataset:
    """Charge le split et calcule les vitesses instantanées jointes aux positions (40, 390).

    Args:
        processed_dir: Répertoire contenant les fichiers préprocessés.
        split_name: Nom du split ('train', 'val', 'test').
        augment: Si True, active l'augmentation aléatoire.

    Returns:
        SignPoseDataset avec 390 caractéristiques par trame.
    """
    base_dataset = load_split_dataset(processed_dir, split_name, augment=False)
    positions = np.array(base_dataset._features, copy=True)
    velocities = np.zeros_like(positions)
    velocities[:, 1:, :] = positions[:, 1:, :] - positions[:, :-1, :]
    enriched = np.concatenate([positions, velocities], axis=-1).astype(np.float32)
    return SignPoseDataset(enriched, base_dataset._labels, augment=augment)


