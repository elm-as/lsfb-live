# 002 — Architecture par domaine

**Date :** 2026-11-06
**Statut :** acceptée

## Décision

Le code vit dans `src/lsfb_live/`, découpé par domaine :

| Domaine | Responsabilité |
|---|---|
| `dataset/` | Téléchargement LSFB ISOL, index des instances, sélection des signes retenus |
| `landmarks/` | Sélection des points et normalisation des poses |
| `recognition/` | Modèles, entraînement, évaluation, inférence |
| `capture/` | Webcam et extraction MediaPipe temps réel |
| `display/` | Rendu à l'écran |

Flux de dépendances : `capture` et `dataset` → `landmarks` → `recognition` → `display`.
Aucun domaine n'importe un domaine situé après lui dans ce flux.

## Règle critique

`landmarks/` est l'**unique** implémentation de la normalisation. Les données d'entraînement
et le flux webcam passent par le même code : toute divergence ferait chuter le modèle en live
sans que les métriques de test ne le révèlent.

## Nommage

Modules en `snake_case` : la charte prévoit du `kebab-case` pour les fichiers logiques,
mais Python ne peut pas importer un module dont le nom contient `-`.

