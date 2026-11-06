"""Dataset PyTorch pour les tenseurs de séquences de squelettes LSFB."""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset


def apply_training_augmentation(
    sequence: np.ndarray,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Applique des augmentations légères (bruit gaussien, jitter d'échelle) sur la séquence.

    Args:
        sequence: Tableau float32 de forme (40, 195).
        noise_std: Écart-type du bruit gaussien ajouté aux coordonnées.

    Returns:
        Nouvelle séquence bruitée float32.
    """
    scale_jitter = np.random.uniform(0.95, 1.05)
    noise = np.random.normal(0.0, noise_std, size=sequence.shape).astype(np.float32)
    augmented = (sequence * scale_jitter) + noise
    return augmented.astype(np.float32)


class SignPoseDataset(Dataset):
    """Dataset PyTorch gérant les tenseurs de poses et leurs étiquettes de classe.

    Args:
        features: Tableau de forme (N, 40, 195).
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
        sample = self._features[index]
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

