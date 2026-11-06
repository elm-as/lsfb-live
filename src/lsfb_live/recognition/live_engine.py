"""Moteur d'inférence en direct par fenêtre glissante et lissage temporel."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import numpy as np
import torch
import torch.nn.functional as F

from lsfb_live.capture.holistic_extractor import FrameLandmarks
from lsfb_live.landmarks import (
    TARGET_SEQUENCE_LENGTH,
    transform_enriched_landmarks_pipeline,
    transform_landmarks_pipeline,
)
from lsfb_live.recognition.models import (
    BiGRUClassifier,
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
        model_type: str = "bigru",
        window_size: int = 40,
        confidence_threshold: float = 0.45,
    ):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            self._class_to_sign = mapping_data["class_to_sign"]

        num_classes = len(self._class_to_sign)
        self._use_velocity = model_type == "msst"
        input_dim = 390 if self._use_velocity else 195
        config = ModelArchitectureConfig(input_dim=input_dim, num_classes=num_classes)

        if model_type == "bigru":
            self._model = BiGRUClassifier(config)
        elif model_type == "transformer":
            self._model = ConvTransformerClassifier(config)
        elif model_type == "msst":
            self._model = MSSTClassifier(config)
        else:
            raise ValueError(f"Type de modèle inconnu : {model_type}")

        self._model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self._model.eval()

        self._window_size = window_size
        self._threshold = confidence_threshold

        self._pose_history: deque[np.ndarray] = deque(maxlen=window_size)
        self._lh_history: deque[np.ndarray] = deque(maxlen=window_size)
        self._rh_history: deque[np.ndarray] = deque(maxlen=window_size)
        self._smoothed_probs: Optional[np.ndarray] = None

    def push_frame(self, frame_landmarks: FrameLandmarks) -> Optional[SignPrediction]:
        """Ajoute une trame de repères au buffer et calcule la prédiction si assez d'historique.

        Args:
            frame_landmarks: Repères extraits de la trame courante.

        Returns:
            SignPrediction ou None si le buffer contient moins de 15 trames.
        """
        self._pose_history.append(frame_landmarks.pose)
        self._lh_history.append(frame_landmarks.left_hand)
        self._rh_history.append(frame_landmarks.right_hand)

        # Minimum de trames pour avoir un geste significatif
        if len(self._pose_history) < 15:
            return None

        # Si aucune main n'est visible sur les 5 dernières trames, l'utilisateur ne signe pas
        recent_has_hands = any(
            not np.all(self._lh_history[i] == 0) or not np.all(self._rh_history[i] == 0)
            for i in range(-min(5, len(self._pose_history)), 0)
        )
        if not recent_has_hands:
            self._smoothed_probs = None
            return SignPrediction(sign="En attente...", confidence=0.0, top5=[], is_active_sign=False)

        pose_seq = np.array(self._pose_history)
        lh_seq = np.array(self._lh_history)
        rh_seq = np.array(self._rh_history)

        # Détection d'inactivité : si les poignets sont immobiles, on évite les fausses prédictions
        wrist_motion = (
            np.sum(np.abs(np.diff(pose_seq[-10:, 15, :2], axis=0)))
            + np.sum(np.abs(np.diff(pose_seq[-10:, 16, :2], axis=0)))
        )
        if wrist_motion < 0.05:
            return SignPrediction(sign="En attente d'un geste...", confidence=0.0, top5=[], is_active_sign=False)

        # Utilisation stricte de la chaîne de normalisation partagée train/live
        if self._use_velocity:
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

