"""Tests unitaires pour le moteur d'inférence temps réel sur fenêtre glissante."""

from pathlib import Path
import json
import numpy as np
import pytest
import torch

from lsfb_live.capture.holistic_extractor import FrameLandmarks
from lsfb_live.recognition.live_engine import LiveInferenceEngine
from lsfb_live.recognition.models import BiGRUClassifier, ModelArchitectureConfig


@pytest.fixture
def mock_environment(tmp_path: Path):
    """Crée des fichiers factices de poids et de mappage pour tester l'inférence."""
    mapping = {
        "sign_to_class": {"AUSSI": 0, "OUI": 1},
        "class_to_sign": ["AUSSI", "OUI"],
    }
    mapping_path = tmp_path / "mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)

    config = ModelArchitectureConfig(num_classes=2)
    model = BiGRUClassifier(config)
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    return model_path, mapping_path


def test_live_engine_buffers_and_predicts(mock_environment):
    """Vérifie que l'inférence retourne None sous 15 trames, puis une prédiction valide."""
    model_path, mapping_path = mock_environment
    engine = LiveInferenceEngine(
        model_path=model_path,
        mapping_path=mapping_path,
        model_type="bigru",
        window_size=40,
        confidence_threshold=0.3,
    )

    dummy_landmarks = FrameLandmarks(
        pose=np.ones((33, 3), dtype=np.float32),
        left_hand=np.ones((21, 3), dtype=np.float32),
        right_hand=np.ones((21, 3), dtype=np.float32),
        has_pose=True,
        has_hands=True,
    )

    # 10 trames : historique insuffisant -> None
    for _ in range(10):
        pred = engine.push_frame(dummy_landmarks)
        assert pred is None

    # Ajout jusqu'à 20 trames avec mouvement actif des mains -> prédiction active
    last_pred = None
    for step in range(10):
        moving_pose = np.ones((33, 3), dtype=np.float32)
        moving_pose[15, :2] += float(step) * 0.05
        moving_pose[16, :2] += float(step) * 0.05
        active_landmarks = FrameLandmarks(
            pose=moving_pose,
            left_hand=np.ones((21, 3), dtype=np.float32),
            right_hand=np.ones((21, 3), dtype=np.float32),
            has_pose=True,
            has_hands=True,
        )
        last_pred = engine.push_frame(active_landmarks)

    assert last_pred is not None
    assert last_pred.confidence >= 0.0
    assert len(last_pred.top5) == 2

