"""Moteur d'inférence en direct par fenêtre glissante et lissage temporel."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from lsfb_live.capture.holistic_extractor import FrameLandmarks
from lsfb_live.landmarks import (
    TARGET_SEQUENCE_LENGTH,
    assemble_bistream_features,
    transform_enriched_landmarks_pipeline,
    transform_landmarks_pipeline,
)
from lsfb_live.recognition.models import (
    BiGRUClassifier,
    BiStreamMSSTClassifier,
    ConvTransformerClassifier,
    ModelArchitectureConfig,
    MSSTClassifier,
)


@dataclass(frozen=True)
class SignPrediction:
    """Résultat d'une prédiction sur une fenêtre glissante.

    Attributes:
        sign: Glose textuelle du signe détecté (ou '...' si incertain).
        confidence: Probabilité du signe dominant (0.0 à 1.0).
        top5: Liste des 5 meilleurs candidats avec leur probabilité [(glose, proba)].
        is_active_sign: True si la confiance dépasse le seuil minimal de détection.
    """

    sign: str
    confidence: float
    top5: list[tuple[str, float]]
    is_active_sign: bool


class LiveInferenceEngine:
    """Orchestre la fenêtre temporelle glissante et l'inférence du réseau de neurones.

    Args:
        model_path: Fichier des poids PyTorch (.pt).
        mapping_path: Fichier JSON contenant class_to_sign.
        model_type: 'bigru' ou 'transformer'.
        window_size: Nombre de trames conservées dans la mémoire glissante (40).
        confidence_threshold: Seuil minimal de confiance pour valider un signe (0.45).
    """

    def __init__(
        self,
        model_path: Path,
        mapping_path: Path,
        model_type: str = "bistream",
        window_size: int = 40,
        confidence_threshold: float = 0.45,
    ):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            self._class_to_sign = mapping_data["class_to_sign"]

        num_classes = len(self._class_to_sign)
        self._model_type = model_type
        self._use_velocity = model_type == "msst"

        if model_type == "bistream":
            config = ModelArchitectureConfig(input_dim=416, hidden_dim=192, num_classes=num_classes)
            self._model = BiStreamMSSTClassifier(config)
        elif model_type == "msst":
            config = ModelArchitectureConfig(input_dim=390, num_classes=num_classes)
            self._model = MSSTClassifier(config)
        elif model_type == "transformer":
            config = ModelArchitectureConfig(input_dim=195, num_classes=num_classes)
            self._model = ConvTransformerClassifier(config)
        elif model_type == "bigru":
            config = ModelArchitectureConfig(input_dim=195, num_classes=num_classes)
            self._model = BiGRUClassifier(config)
        else:
            raise ValueError(f"Type de modèle inconnu : {model_type}")

        self._model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self._model.eval()

        self._window_size = window_size
        self._threshold = confidence_threshold
        self._window_duration = 1.3  # Durée canonique d'un signe en secondes

        self._history: deque[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=120)
        self._smoothed_probs: Optional[np.ndarray] = None

    def push_frame(self, frame_landmarks: FrameLandmarks) -> Optional[SignPrediction]:
        """Ajoute une trame de repères au buffer et calcule la prédiction si assez d'historique.

        Args:
            frame_landmarks: Repères extraits de la trame courante.

        Returns:
            SignPrediction ou None si le buffer contient moins de trames significatives.
        """
        now = time.time()
        self._history.append((now, frame_landmarks.pose, frame_landmarks.left_hand, frame_landmarks.right_hand))

        # Fenêtre temporelle glissante invariante au framerate (dernières 1.3 secondes)
        cutoff = now - self._window_duration
        recent = [item for item in self._history if item[0] >= cutoff]

        # Minimum de trames pour interpoler un geste
        if len(recent) < 12:
            return None

        # Si aucune main n'est visible sur les dernières trames, repos
        has_hands_recent = any(
            not np.all(item[2] == 0) or not np.all(item[3] == 0)
            for item in recent[-4:]
        )
        if not has_hands_recent:
            self._smoothed_probs = None
            return SignPrediction(sign="En attente...", confidence=0.0, top5=[], is_active_sign=False)

        pose_seq = np.array([item[1] for item in recent])
        lh_seq = np.array([item[2] for item in recent])
        rh_seq = np.array([item[3] for item in recent])

        # Détection d'inactivité : poignets et pointes d'index immobiles
        wrist_motion = (
            np.sum(np.abs(np.diff(pose_seq[:, 15, :2], axis=0)))
            + np.sum(np.abs(np.diff(pose_seq[:, 16, :2], axis=0)))
            + np.sum(np.abs(np.diff(lh_seq[:, 8, :2], axis=0)))
            + np.sum(np.abs(np.diff(rh_seq[:, 8, :2], axis=0)))
        )
        if wrist_motion < 0.03:
            return SignPrediction(sign="En attente d'un geste...", confidence=0.0, top5=[], is_active_sign=False)

        # Utilisation stricte de la chaîne de normalisation partagée train/live
        if self._model_type == "bistream":
            positions = transform_landmarks_pipeline(pose_seq, lh_seq, rh_seq)
            features = assemble_bistream_features(positions)
        elif self._use_velocity:
            features = transform_enriched_landmarks_pipeline(pose_seq, lh_seq, rh_seq)
        else:
            features = transform_landmarks_pipeline(pose_seq, lh_seq, rh_seq)

        tensor_x = torch.from_numpy(features).unsqueeze(0).float()  # (1, 40, input_dim)

        with torch.no_grad():
            logits = self._model(tensor_x)
            probabilities = F.softmax(logits, dim=1).squeeze(0).numpy()

        # Lissage exponentiel des probabilités (alpha = 0.4)
        if self._smoothed_probs is None:
            self._smoothed_probs = probabilities
        else:
            self._smoothed_probs = 0.6 * self._smoothed_probs + 0.4 * probabilities

        top_indices = np.argsort(self._smoothed_probs)[::-1][:5]
        top5 = [(self._class_to_sign[idx], float(self._smoothed_probs[idx])) for idx in top_indices]

        best_sign, best_conf = top5[0]
        is_active = best_conf >= self._threshold

        return SignPrediction(
            sign=best_sign if is_active else f"({best_sign}?)",
            confidence=best_conf,
            top5=top5,
            is_active_sign=is_active,
        )

