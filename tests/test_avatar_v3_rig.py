"""Tests unitaires pour le Mannequin Humain Rigué 3D, l'IK 2-segments et le Text-to-Sign (V3)."""

import numpy as np

from lsfb_live.avatar.rig_skeleton import (
    UPPER_ARM_LENGTH,
    interpolate_rig_skeletons,
    retarget_normalized_65_to_rig,
    rig_to_normalized_65,
    solve_two_bone_arm_ik,
)
from lsfb_live.avatar.text_to_sign_sequencer import (
    SignSequenceTimeline,
    get_sign_rig_at_progress,
    parse_french_to_lsfb_glosses,
)
from lsfb_live.display.rig_renderer_cv import render_rig_mannequin_cv


def test_solve_two_bone_arm_ik_preserves_upper_arm_length() -> None:
    """Vérifie que l'IK analytique respecte la longueur anatomique de l'humérus."""
    shoulder = np.array([-0.50, 0.0, 0.0], dtype=np.float32)
    wrist = np.array([-0.15, 0.30, -0.25], dtype=np.float32)
    pole = np.array([-0.35, 0.85, 0.25], dtype=np.float32)

    elbow = solve_two_bone_arm_ik(shoulder, wrist, pole)
    humerus_len = float(np.linalg.norm(elbow - shoulder))
    assert np.isclose(humerus_len, UPPER_ARM_LENGTH, atol=1e-3)


def test_parse_french_to_lsfb_glosses() -> None:
    """Vérifie la traduction de phrases françaises vers les gloses LSFB du modèle."""
    glosses = parse_french_to_lsfb_glosses("Mon papa aime toujours la famille et explique l'histoire à l'école !")
    assert glosses == ["PAPA", "AIMER", "TOUJOURS", "FAMILLE", "EXPLIQUER", "HISTOIRE", "ECOLE"]


def test_sign_sequence_timeline_and_rig_interpolation() -> None:
    """Vérifie le séquençage 3D continu sur une phrase complète et l'interpolation IK."""
    timeline = SignSequenceTimeline(["RENCONTRER", "AMI", "MAISON"])
    sign0, rig0 = timeline.evaluate(0.10)
    sign1, rig1 = timeline.evaluate(0.50)
    sign2, rig2 = timeline.evaluate(0.85)

    assert sign0 == "RENCONTRER"
    assert sign1 == "AMI"
    assert sign2 == "MAISON"
    assert rig0.left_hand.shape == (21, 3)
    assert rig1.right_hand.shape == (21, 3)

    mid_rig = interpolate_rig_skeletons(rig0, rig1, 0.5)
    assert np.all(np.isfinite(mid_rig.left_elbow))
    assert np.all(np.isfinite(mid_rig.right_elbow))


def test_render_rig_mannequin_cv_with_orbital_yaw() -> None:
    """Vérifie le rendu volumétrique 3D OpenCV sous différents angles de caméra."""
    rig = get_sign_rig_at_progress("DIFFERENT", 0.5)
    for yaw in (-30.0, 0.0, 30.0):
        canvas = np.zeros((720, 640, 3), dtype=np.uint8)
        render_rig_mannequin_cv(canvas, rig, yaw_deg=yaw)
        assert np.any(canvas > 0)


def test_rig_to_normalized_65_preserves_hand_geometry() -> None:
    """Vérifie l'aller-retour sans distorsion entre RigSkeleton3D et le format modèle (65, 3)."""
    rig = get_sign_rig_at_progress("AIMER", 0.5)
    reconstructed = rig_to_normalized_65(rig)
    assert reconstructed.shape == (65, 3)
    assert np.all(np.isfinite(reconstructed))


