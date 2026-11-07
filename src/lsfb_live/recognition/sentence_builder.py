"""Accumulateur de gloses et constructeur de phrases LSFB en flux continu (Base v2.0).

Transforme une suite d'inférences temps réel bruitées en une séquence propre de gloses
validées par stabilité temporelle, accompagnée d'une transcription lisible en français.
"""

from dataclasses import dataclass
from typing import Optional

from lsfb_live.recognition.live_engine import SignPrediction

# Dictionnaire de transcription lexicale LSFB -> Français naturel
LEXICAL_FRENCH_MAP: dict[str, str] = {
    "ADORER": "adore",
    "AIDER": "aide",
    "AIMER": "aime",
    "AMI": "un ami",
    "APPRENDRE": "apprend",
    "AUJOURD-HUI": "aujourd'hui",
    "AVANCER": "avance",
    "CHERCHER": "cherche",
    "COMMENCER": "commence",
    "COMPRENDRE": "comprend",
    "CONVENIR": "convient",
    "CORPS": "le corps",
    "DIFFERENT": "différent",
    "ECOLE": "à l'école",
    "ENTENDANT": "entendant",
    "EXPLIQUER": "explique",
    "FAMILLE": "la famille",
    "FINIR": "termine",
    "HISTOIRE": "l'histoire",
    "INTERPRETE": "l'interprète",
    "MAISON": "à la maison",
    "MANGER": "mange",
    "MONTRER": "montre",
    "OUBLIER": "oublie",
    "PAPA": "papa",
    "PARLER": "parle",
    "REGARDER": "regarde",
    "RENCONTRER": "rencontre",
    "SOURD": "sourd",
    "TOUJOURS": "toujours",
}


@dataclass(frozen=True)
class SentenceSnapshot:
    """État immuable de la phrase en cours de construction en direct.

    Attributes:
        glosses: Liste ordonnée des gloses LSFB validées.
        french_text: Transcription fluide en français.
        candidate_sign: Signe en cours de stabilisation (avant validation).
        stability_ratio: Progression de stabilisation du signe candidat (0.0 à 1.0).
    """

    glosses: tuple[str, ...]
    french_text: str
    candidate_sign: Optional[str]
    stability_ratio: float


def format_gloss_sequence_to_french(glosses: tuple[str, ...]) -> str:
    """Convertit un tuple de gloses LSFB en une phrase française lisible (fonction pure).

    Args:
        glosses: Séquence immuable de gloses LSFB (ex: ('PAPA', 'AIMER', 'FAMILLE')).

    Returns:
        Chaîne formatée en français avec majuscule initiale.
    """
    if not glosses:
        return "Signez face a la camera pour composer une phrase..."

    words = [LEXICAL_FRENCH_MAP.get(gloss, gloss.lower()) for gloss in glosses]
    sentence = " ".join(words).strip()
    if not sentence:
        return ""
    return sentence[0].upper() + sentence[1:] + "."


class GlossSentenceAccumulator:
    """Accumulateur temporel filtrant les prédictions en direct pour former une phrase."""

    def __init__(self, min_stable_steps: int = 3, max_glosses: int = 7) -> None:
        if min_stable_steps < 1:
            raise ValueError("min_stable_steps doit être supérieur ou égal à 1.")
        self._min_stable_steps = min_stable_steps
        self._max_glosses = max_glosses
        self._glosses: list[str] = []
        self._current_candidate: Optional[str] = None
        self._candidate_streak: int = 0
        self._last_committed_sign: Optional[str] = None

    def push_prediction(self, prediction: Optional[SignPrediction]) -> SentenceSnapshot:
        """Intègre une nouvelle prédiction et retourne l'état courant de la phrase."""
        if prediction is None or not prediction.is_active_sign:
            self._current_candidate = None
            self._candidate_streak = 0
            if prediction is not None and prediction.sign == "[Repos]":
                self._last_committed_sign = None
            return self.get_snapshot()

        sign = prediction.sign
        if sign == self._last_committed_sign:
            return self.get_snapshot()

        if sign == self._current_candidate:
            self._candidate_streak += 1
        else:
            self._current_candidate = sign
            self._candidate_streak = 1

        if self._candidate_streak >= self._min_stable_steps:
            self._commit_sign(sign)

        return self.get_snapshot()

    def _commit_sign(self, sign: str) -> None:
        """Ajoute un signe validé à la phrase et réinitialise le compteur de stabilité."""
        self._glosses.append(sign)
        if len(self._glosses) > self._max_glosses:
            self._glosses = self._glosses[-self._max_glosses :]
        self._last_committed_sign = sign
        self._current_candidate = None
        self._candidate_streak = 0

    def pop_last(self) -> SentenceSnapshot:
        """Supprime la dernière glose validée (touche Retour arrière)."""
        if self._glosses:
            self._glosses.pop()
        self._last_committed_sign = self._glosses[-1] if self._glosses else None
        return self.get_snapshot()

    def clear(self) -> SentenceSnapshot:
        """Vide intégralement la phrase en cours (touche C)."""
        self._glosses.clear()
        self._current_candidate = None
        self._candidate_streak = 0
        self._last_committed_sign = None
        return self.get_snapshot()

    def get_snapshot(self) -> SentenceSnapshot:
        """Retourne une vue immuable de l'état courant de la phrase."""
        frozen_glosses = tuple(self._glosses)
        ratio = float(min(1.0, self._candidate_streak / float(self._min_stable_steps)))
        return SentenceSnapshot(
            glosses=frozen_glosses,
            french_text=format_gloss_sequence_to_french(frozen_glosses),
            candidate_sign=self._current_candidate,
            stability_ratio=ratio,
        )
