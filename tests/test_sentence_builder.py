"""Tests unitaires pour l'accumulateur de phrases LSFB en direct (v2.0)."""

from lsfb_live.recognition.live_engine import SignPrediction
from lsfb_live.recognition.sentence_builder import (
    GlossSentenceAccumulator,
    format_gloss_sequence_to_french,
)


def _make_pred(sign: str, conf: float = 0.75, active: bool = True) -> SignPrediction:
    return SignPrediction(
        sign=sign,
        confidence=conf,
        top5=[(sign, conf)],
        is_active_sign=active,
    )


def test_format_gloss_sequence_to_french() -> None:
    """Vérifie la transcription pure d'une suite de gloses LSFB en français."""
    text = format_gloss_sequence_to_french(("PAPA", "AIMER", "FAMILLE"))
    assert text == "Papa aime la famille."


def test_sentence_accumulator_requires_stability_and_deduplicates() -> None:
    """Vérifie qu'un signe doit être stable sur N étapes et n'est pas dupliqué en continu."""
    acc = GlossSentenceAccumulator(min_stable_steps=3, max_glosses=5)

    # 2 étapes seulement : pas encore validé
    acc.push_prediction(_make_pred("PAPA"))
    snap = acc.push_prediction(_make_pred("PAPA"))
    assert snap.glosses == ()
    assert snap.candidate_sign == "PAPA"

    # 3ème étape consécutive : validé !
    snap = acc.push_prediction(_make_pred("PAPA"))
    assert snap.glosses == ("PAPA",)

    # Maintenir PAPA ne le duplique pas
    snap = acc.push_prediction(_make_pred("PAPA"))
    assert snap.glosses == ("PAPA",)

    # Nouveau signe AIMER sur 3 étapes
    for _ in range(3):
        snap = acc.push_prediction(_make_pred("AIMER"))
    assert snap.glosses == ("PAPA", "AIMER")
    assert snap.french_text == "Papa aime."

    # Suppression du dernier mot et remise à zéro
    snap = acc.pop_last()
    assert snap.glosses == ("PAPA",)
    snap = acc.clear()
    assert snap.glosses == ()

