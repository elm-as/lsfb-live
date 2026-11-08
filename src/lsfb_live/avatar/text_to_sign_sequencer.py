"""Moteur Text-to-Sign (V3) : lemmatisation Français -> Gloses LSFB et séquençage 3D."""

from pathlib import Path
import json
import re
import unicodedata
import numpy as np

from lsfb_live.avatar.rig_skeleton import (
    RigSkeleton3D,
    build_relaxed_hand_3d,
    interpolate_rig_skeletons,
    retarget_normalized_65_to_rig,
    solve_two_bone_arm_ik,
)

# Table de correspondance racinisée Français -> 30 Gloses LSFB du modèle v1.5
_FRENCH_STEM_TO_GLOSS: dict[str, str] = {
    "ador": "ADORER", "adore": "ADORER", "adorer": "ADORER",
    "aid": "AIDER", "aide": "AIDER", "aider": "AIDER",
    "aim": "AIMER", "aime": "AIMER", "aimer": "AIMER", "amour": "AIMER",
    "ami": "AMI", "amie": "AMI", "amis": "AMI", "copain": "AMI",
    "apprend": "APPRENDRE", "apprendre": "APPRENDRE", "appris": "APPRENDRE", "etudi": "APPRENDRE",
    "aujourd": "AUJOURD-HUI", "aujourdhui": "AUJOURD-HUI", "maintenant": "AUJOURD-HUI",
    "avanc": "AVANCER", "avance": "AVANCER", "avancer": "AVANCER",
    "cherch": "CHERCHER", "cherche": "CHERCHER", "chercher": "CHERCHER",
    "commenc": "COMMENCER", "commence": "COMMENCER", "commencer": "COMMENCER", "debut": "COMMENCER",
    "comprend": "COMPRENDRE", "comprendre": "COMPRENDRE", "compris": "COMPRENDRE",
    "convien": "CONVENIR", "convenir": "CONVENIR", "accord": "CONVENIR", "ok": "CONVENIR",
    "corps": "CORPS", "physique": "CORPS", "buste": "CORPS",
    "different": "DIFFERENT", "differente": "DIFFERENT", "differents": "DIFFERENT", "autre": "DIFFERENT",
    "ecol": "ECOLE", "ecole": "ECOLE", "classe": "ECOLE", "cours": "ECOLE", "universite": "ECOLE",
    "entend": "ENTENDANT", "entendant": "ENTENDANT", "entendre": "ENTENDANT",
    "expliqu": "EXPLIQUER", "explique": "EXPLIQUER", "expliquer": "EXPLIQUER",
    "famill": "FAMILLE", "famille": "FAMILLE", "parent": "FAMILLE", "parents": "FAMILLE",
    "fini": "FINIR", "finir": "FINIR", "termin": "FINIR", "termine": "FINIR", "fin": "FINIR",
    "histoir": "HISTOIRE", "histoire": "HISTOIRE", "conte": "HISTOIRE", "racont": "HISTOIRE",
    "interpret": "INTERPRETE", "interprete": "INTERPRETE", "tradui": "INTERPRETE", "traducteur": "INTERPRETE",
    "maison": "MAISON", "foyer": "MAISON", "chez": "MAISON", "domicile": "MAISON",
    "mang": "MANGER", "mange": "MANGER", "manger": "MANGER", "repas": "MANGER",
    "montr": "MONTRER", "montre": "MONTRER", "montrer": "MONTRER", "voici": "MONTRER",
    "oubli": "OUBLIER", "oublie": "OUBLIER", "oublier": "OUBLIER",
    "papa": "PAPA", "pere": "PAPA", "dar": "PAPA",
    "parl": "PARLER", "parle": "PARLER", "parler": "PARLER", "dire": "PARLER", "dis": "PARLER",
    "regard": "REGARDER", "regarde": "REGARDER", "regarder": "REGARDER", "voir": "REGARDER", "vois": "REGARDER",
    "rencontr": "RENCONTRER", "rencontre": "RENCONTRER", "rencontrer": "RENCONTRER", "retrouv": "RENCONTRER",
    "sourd": "SOURD", "sourde": "SOURD", "sourds": "SOURD", "malentendant": "SOURD",
    "toujour": "TOUJOURS", "toujours": "TOUJOURS", "encore": "TOUJOURS", "jamais": "TOUJOURS",
}

_PROTOTYPES_CACHE: dict[str, np.ndarray] = {}
_PROTOTYPES_CONF_CACHE: dict[str, float] = {}


def _strip_accents(text: str) -> str:
    """Normalise une chaîne en minuscules sans accents diacritiques."""
    norm = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def parse_french_to_lsfb_glosses(text: str) -> list[str]:
    """Traduit une phrase en français naturel en liste ordonnée de gloses LSFB.

    Args:
        text: Phrase en français ou suite de gloses (ex: "Mon papa aime la famille").

    Returns:
        Liste des gloses LSFB reconnues (ex: ['PAPA', 'AIMER', 'FAMILLE']).
    """
    clean = _strip_accents(text.replace("aujourd'hui", "aujourdhui").replace("aujourd hui", "aujourdhui"))
    tokens = re.findall(r"[a-z]+", clean)
    glosses: list[str] = []

    for token in tokens:
        upper = token.upper()
        if upper == "AUJOURDHUI":
            glosses.append("AUJOURD-HUI")
            continue
        matched = _FRENCH_STEM_TO_GLOSS.get(token)
        if matched is None:
            for prefix_len in (7, 6, 5, 4):
                if len(token) >= prefix_len and token[:prefix_len] in _FRENCH_STEM_TO_GLOSS:
                    matched = _FRENCH_STEM_TO_GLOSS[token[:prefix_len]]
                    break
        if matched is not None and (not glosses or glosses[-1] != matched):
            glosses.append(matched)

    return glosses


def load_sign_prototypes_3d(
    json_path: Path = Path("models/sign_prototypes_3d.json"),
) -> dict[str, np.ndarray]:
    """Charge et met en cache les trajectoires 3D (24, 65, 3) des 30 signes LSFB."""
    if _PROTOTYPES_CACHE:
        return _PROTOTYPES_CACHE
    if not json_path.is_file():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for sign_name, payload in raw.items():
        _PROTOTYPES_CACHE[sign_name] = np.array(payload["frames"], dtype=np.float32)
        _PROTOTYPES_CONF_CACHE[sign_name] = float(payload.get("model_confidence", 0.90))
    return _PROTOTYPES_CACHE


def get_prototype_model_confidence(sign_name: str) -> float:
    """Retourne le score de reconnaissance du modèle BiStream-MSST sur le mouvement du mannequin."""
    load_sign_prototypes_3d()
    return _PROTOTYPES_CONF_CACHE.get(sign_name, 0.90)


def _build_procedural_fallback_rig(sign_name: str, progress: float) -> RigSkeleton3D:
    """Génère un rig 3D procédural si le fichier de prototypes n'est pas encore chargé."""
    t = float(np.clip(progress, 0.0, 1.0))
    smooth = float(0.5 - 0.5 * np.cos(np.pi * t))
    ls = np.array([-0.50, 0.0, 0.0], dtype=np.float32)
    rs = np.array([0.50, 0.0, 0.0], dtype=np.float32)

    if sign_name == "DIFFERENT":
        lw = np.array([-0.05 - 0.40 * smooth, 0.25, -0.25], dtype=np.float32)
        rw = np.array([0.05 + 0.40 * smooth, 0.25, -0.25], dtype=np.float32)
    elif sign_name == "RENCONTRER":
        lw = np.array([-0.42 + 0.34 * smooth, 0.25, -0.25], dtype=np.float32)
        rw = np.array([0.42 - 0.34 * smooth, 0.25, -0.25], dtype=np.float32)
    else:
        lw = np.array([-0.38, 0.72, -0.10], dtype=np.float32)
        rw = np.array([0.15, 0.20 + 0.15 * smooth, -0.24], dtype=np.float32)

    le = solve_two_bone_arm_ik(ls, lw, np.array([-0.35, 0.85, 0.25], dtype=np.float32))
    re = solve_two_bone_arm_ik(rs, rw, np.array([0.35, 0.85, 0.25], dtype=np.float32))
    return RigSkeleton3D(
        head_center=np.array([0.0, -0.58, -0.05], dtype=np.float32),
        nose_tip=np.array([0.0, -0.55, -0.20], dtype=np.float32),
        neck=np.array([0.0, -0.14, 0.0], dtype=np.float32),
        pelvis=np.array([0.0, 0.92, 0.04], dtype=np.float32),
        left_shoulder=ls,
        right_shoulder=rs,
        left_elbow=le,
        right_elbow=re,
        left_wrist=lw,
        right_wrist=rw,
        left_hand=build_relaxed_hand_3d(lw, is_mirror_left=True),
        right_hand=build_relaxed_hand_3d(rw, is_mirror_left=False),
    )


def get_sign_rig_at_progress(sign_name: str, progress: float) -> RigSkeleton3D:
    """Retourne le RigSkeleton3D interpolé à l'instant `progress` (0.0 à 1.0) pour un signe."""
    prototypes = load_sign_prototypes_3d()
    if sign_name not in prototypes:
        return _build_procedural_fallback_rig(sign_name, progress)

    seq = prototypes[sign_name]  # (K, 65, 3)
    num_frames = len(seq)
    pos = float(np.clip(progress, 0.0, 1.0)) * (num_frames - 1)
    idx0 = int(pos)
    idx1 = min(num_frames - 1, idx0 + 1)
    frac = pos - idx0

    frame_65 = (1.0 - frac) * seq[idx0] + frac * seq[idx1]
    return retarget_normalized_65_to_rig(frame_65)


class SignSequenceTimeline:
    """Séquenceur temporel enchaînant une liste de gloses LSFB avec transitions 3D fluides."""

    def __init__(self, glosses: list[str]) -> None:
        self._glosses = glosses if glosses else ["AIMER"]

    @property
    def glosses(self) -> list[str]:
        return list(self._glosses)

    def evaluate(self, global_progress: float) -> tuple[str, RigSkeleton3D]:
        """Calcule la glose courante et l'état 3D du rig humain pour `global_progress` dans [0, 1]."""
        clamped = float(np.clip(global_progress, 0.0, 0.9999))
        scaled = clamped * len(self._glosses)
        idx = int(scaled)
        local_t = scaled - idx
        curr_sign = self._glosses[idx]

        # 80% exécution du signe natif, 20% transition IK cosinus vers le signe suivant
        if local_t <= 0.80 or len(self._glosses) == 1:
            sign_t = min(1.0, local_t / 0.80) if len(self._glosses) > 1 else local_t
            return curr_sign, get_sign_rig_at_progress(curr_sign, sign_t)

        next_sign = self._glosses[(idx + 1) % len(self._glosses)]
        blend = (local_t - 0.80) / 0.20
        rig_end = get_sign_rig_at_progress(curr_sign, 1.0)
        rig_next = get_sign_rig_at_progress(next_sign, 0.0)
        return curr_sign, interpolate_rig_skeletons(rig_end, rig_next, blend)

