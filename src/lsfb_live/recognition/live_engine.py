"""Moteur d'inférence en direct multi-échelles temporelles avec découpage d'onset gestuel."""

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


def _trim_active_gesture_frames(
    recent: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Découpe la fenêtre temporelle dès l'entrée effective des mains dans le champ.

    Pour un signe bimanuel (ex: RENCONTRER, ADORER, DIFFERENT), aligne la trame 0
    sur l'instant où les deux mains sont apparues afin de capturer 100% du déplacement.
    """
    active = [item for item in recent if not np.all(item[2] == 0) or not np.all(item[3] == 0)]
    if len(active) < 5:
        return recent

    # Si l'utilisateur signe actuellement à 2 mains, on cale le début sur l'apparition des 2 mains
    both_hands_now = sum(
        (not np.all(item[2] == 0) and not np.all(item[3] == 0))
        for item in active[-3:]
    ) >= 2
    if both_hands_now:
        for idx, item in enumerate(active):
            if not np.all(item[2] == 0) and not np.all(item[3] == 0):
                if len(active) - idx >= 5:
                    return active[idx:]
                break
    return active


def _extract_motion_core_frames(
    active: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
) -> Optional[list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]]:
    """Isole la phase de mouvement actif en retirant le maintien immobile final.

    Lorsqu'un signeur exécute un signe à déplacement (ex: RENCONTRER, ADORER) puis
    garde les mains immobiles pour lire l'écran, cette fonction récupère l'arc
    cinématique complet [départ -> mouvement -> arrivée] avant l'immobilisation.
    """
    if len(active) < 9:
        return None

    end_idx = len(active)
    while end_idx > 6:
        curr = active[end_idx - 1]
        prev = active[end_idx - 2]
        dl = float(np.linalg.norm(curr[2][0, :2] - prev[2][0, :2])) if not np.all(curr[2] == 0) else 0.0
        dr = float(np.linalg.norm(curr[3][0, :2] - prev[3][0, :2])) if not np.all(curr[3] == 0) else 0.0
        if max(dl, dr) < 0.012:
            end_idx -= 1
        else:
            break

    # On conserve 2 trames de la pose finale pour stabiliser l'arrivée du geste
    trimmed_end = min(len(active), end_idx + 2)
    if trimmed_end < len(active) - 1 and trimmed_end >= 6:
        return active[:trimmed_end]
    return None


class LiveInferenceEngine:
    """Orchestre la fenêtre temporelle bi-échelle et l'inférence du réseau de neurones."""

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
        self._window_duration = 1.65  # Fenêtre élargie pour capter les déplacements complets à 10 FPS

        self._history: deque[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=120)
        self._smoothed_probs: Optional[np.ndarray] = None

    def _encode_window(
        self,
        frames: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Transforme une sous-fenêtre de trames en tenseur de caractéristiques (40, D)."""
        pose_seq = np.array([item[1] for item in frames])
        lh_seq = np.array([item[2] for item in frames])
        rh_seq = np.array([item[3] for item in frames])

        if self._model_type == "bistream":
            positions = transform_landmarks_pipeline(pose_seq, lh_seq, rh_seq)
            return assemble_bistream_features(positions)
        if self._use_velocity:
            return transform_enriched_landmarks_pipeline(pose_seq, lh_seq, rh_seq)
        return transform_landmarks_pipeline(pose_seq, lh_seq, rh_seq)

    def push_frame(self, frame_landmarks: FrameLandmarks) -> Optional[SignPrediction]:
        """Ajoute une trame au buffer et évalue le geste à double échelle temporelle."""
        now = time.time()
        self._history.append((now, frame_landmarks.pose, frame_landmarks.left_hand, frame_landmarks.right_hand))

        cutoff = now - self._window_duration
        recent = [item for item in self._history if item[0] >= cutoff]
        if len(recent) < 6 and len(self._history) >= 6:
            recent = list(self._history)[-12:]
        if len(recent) < 6:
            return None

        has_hands_recent = any(
            not np.all(item[2] == 0) or not np.all(item[3] == 0)
            for item in recent[-4:]
        )
        if not has_hands_recent:
            self._smoothed_probs = None
            return SignPrediction(sign="[Repos]", confidence=0.0, top5=[], is_active_sign=False)

        active_frames = _trim_active_gesture_frames(recent)

        # Évaluation tri-échelle : trajectoire complète + cœur de mouvement (sans maintien final) + fenêtre courte
        windows = [self._encode_window(active_frames)]
        motion_core = _extract_motion_core_frames(active_frames)
        if motion_core is not None:
            windows.append(self._encode_window(motion_core))
        if len(active_frames) >= 11:
            short_len = max(6, int(len(active_frames) * 0.60))
            windows.append(self._encode_window(active_frames[-short_len:]))

        tensor_batch = torch.from_numpy(np.stack(windows, axis=0)).float()
        with torch.no_grad():
            logits = self._model(tensor_batch)
            probs_batch = F.softmax(logits, dim=1).numpy()
            probabilities = np.max(probs_batch, axis=0)

        if self._smoothed_probs is None:
            self._smoothed_probs = probabilities
        else:
            self._smoothed_probs = 0.50 * self._smoothed_probs + 0.50 * probabilities

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
