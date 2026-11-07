"""Tests unitaires pour l'avatar cinématique paramétrique et l'interpolateur v3."""

import numpy as np
from lsfb_live.display.parametric_avatar import (
    compute_canonical_sign_pose,
    compute_sequence_avatar_pose,
    interpolate_avatar_poses,
    render_parametric_avatar_frame,
)


def test_compute_canonical_sign_pose_bounds() -> None:
    """Vérifie que les coordonnées cinématiques des 14 signes restent dans [0, 1]."""
    all_signs = [
        "AIMER",
        "PAPA",
        "OUBLIER",
        "COMPRENDRE",
        "RENCONTRER",
        "DIFFERENT",
        "MAISON",
        "CORPS",
        "AVANCER",
        "MONTRER",
        "INTERPRETE",
        "ECOLE",
        "FAMILLE",
        "AMI",
    ]
    for sign in all_signs:
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pose_state = compute_canonical_sign_pose(sign, t)
            lx, ly = pose_state.left_wrist
            rx, ry = pose_state.right_wrist

            assert 0.1 <= lx <= 0.9, f"Left wrist X hors limites pour {sign} à t={t}: {lx}"
            assert 0.1 <= rx <= 0.9, f"Right wrist X hors limites pour {sign} à t={t}: {rx}"
            assert 0.15 <= ly <= 0.95, f"Left wrist Y hors limites pour {sign} à t={t}: {ly}"
            assert 0.15 <= ry <= 0.95, f"Right wrist Y hors limites pour {sign} à t={t}: {ry}"


def test_sequence_avatar_pose_interpolates_smoothly() -> None:
    """Vérifie que le séquenceur Text-to-Sign v3 enchaîne et interpole les signes."""
    sentence = ["PAPA", "AIMER", "FAMILLE"]
    sign_start, pose_start = compute_sequence_avatar_pose(sentence, 0.05)
    sign_mid, pose_mid = compute_sequence_avatar_pose(sentence, 0.50)
    sign_end, pose_end = compute_sequence_avatar_pose(sentence, 0.95)

    assert sign_start == "PAPA"
    assert sign_mid == "AIMER"
    assert sign_end == "FAMILLE"
    assert pose_start.right_hand_shape == "l_shape"
    assert pose_mid.right_hand_shape == "open"

    blended = interpolate_avatar_poses(pose_start, pose_mid, 0.5)
    expected_rx = 0.5 * (pose_start.right_wrist[0] + pose_mid.right_wrist[0])
    assert np.isclose(blended.right_wrist[0], expected_rx, atol=1e-4)


def test_render_parametric_avatar_frame() -> None:
    """Vérifie que le rendu de l'avatar s'exécute sans erreur sur toutes les formes de main."""
    for sign in ["AIMER", "PAPA", "OUBLIER", "MAISON", "DIFFERENT"]:
        canvas = np.zeros((720, 640, 3), dtype=np.uint8)
        render_parametric_avatar_frame(canvas, sign, progress=0.5)
        assert canvas.shape == (720, 640, 3)
        assert np.any(canvas > 0), f"Le canvas doit contenir des pixels tracés pour {sign}."
