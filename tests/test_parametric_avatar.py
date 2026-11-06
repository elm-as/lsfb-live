"""Tests unitaires pour l'avatar cinématique paramétrique."""

import numpy as np
from lsfb_live.display.parametric_avatar import compute_canonical_sign_pose, render_parametric_avatar_frame


def test_compute_canonical_sign_pose_bounds():
    """Vérifie que les coordonnées cinématiques restent dans l'espace [0, 1]."""
    for sign in ["RENCONTRER", "DIFFERENT", "MAISON", "CORPS", "AVANCER", "MONTRER"]:
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pose_state = compute_canonical_sign_pose(sign, t)
            lx, ly = pose_state.left_wrist
            rx, ry = pose_state.right_wrist

            assert 0.1 <= lx <= 0.9, f"Left wrist X hors limites pour {sign} à t={t}: {lx}"
            assert 0.1 <= rx <= 0.9, f"Right wrist X hors limites pour {sign} à t={t}: {rx}"
            assert 0.2 <= ly <= 0.95, f"Left wrist Y hors limites pour {sign} à t={t}: {ly}"
            assert 0.2 <= ry <= 0.95, f"Right wrist Y hors limites pour {sign} à t={t}: {ry}"


def test_render_parametric_avatar_frame():
    """Vérifie que le rendu de l'avatar s'exécute sans erreur et modifie le canvas."""
    canvas = np.zeros((720, 640, 3), dtype=np.uint8)
    render_parametric_avatar_frame(canvas, "DIFFERENT", progress=0.5)

    assert canvas.shape == (720, 640, 3)
    assert np.any(canvas > 0), "Le canvas doit contenir des pixels tracés pour l'avatar."
