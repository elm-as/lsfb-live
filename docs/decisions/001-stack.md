# 001 — Choix de stack

**Date :** 2026-11-06
**Statut :** acceptée

## Contexte

Reconnaissance en direct de signes isolés LSFB, entraînée sur les poses de LSFB ISOL.
Machine cible : laptop Intel i5-8350U, 16 Go RAM, **sans GPU NVIDIA**, webcam HD intégrée.

## Décision

| Brique | Choix | Raison |
|---|---|---|
| Langage | Python 3.12 | Seul écosystème où `lsfb-dataset`, MediaPipe et PyTorch coexistent. 3.12 car MediaPipe ne suit pas les versions récentes de CPython. |
| Données | `lsfb-dataset` | Package officiel : téléchargement filtrable, reprenable, splits par signeur. |
| Extraction live | MediaPipe Holistic | Outil ayant produit les poses du dataset : même topologie de points en entraînement et en live. |
| Modèle | PyTorch (CPU) | Petites architectures séquentielles, contrôle complet, inférence CPU suffisante. |
| Affichage V1 | OpenCV | Déjà requis par la capture webcam ; aucune dépendance supplémentaire. |
| Suivi d'expériences | Fichiers JSON/CSV | Pas de besoin justifiant MLflow ou W&B à cette échelle. |

## Constat à l'installation (2026-11-06)

MediaPipe 1.2.2 ne fournit plus l'ancienne API `mp.solutions.holistic` : seule l'API
`mediapipe.tasks` est disponible. La capture live utilisera donc les *landmarkers* de
cette API. Ils produisent la même topologie de points (33 points de corps, 21 par main),
mais l'équivalence avec les poses du dataset devra être vérifiée à l'étape de capture.

## Conséquences

- Modèles volontairement légers (entraînement CPU, inférence temps réel).
- Un modèle plus lourd nécessiterait un GPU distant (Colab) : à décider seulement si la baseline plafonne.

