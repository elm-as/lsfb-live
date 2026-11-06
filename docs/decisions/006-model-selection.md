# 006 — Sélection du modèle et résultats d'évaluation

**Date :** 2026-11-06  
**Statut :** acceptée

## Contexte

Comparaison de deux architectures légères adaptées à l'inférence temps réel sur CPU (Intel i5)
pour classifier 100 signes LSFB distincts sur 24 749 instances de test issues de 40 signeurs
jamais vus pendant l'entraînement.

## Résultats sur le test officiel (40 signeurs non vus)

| Modèle | Paramètres | Top-1 Accuracy | Top-5 Accuracy | Balanced Accuracy | Test Loss |
|---|---|---|---|---|---|
| **BiGRU (Baseline)** | 320k | 44.80% | 75.76% | 44.66% | 2.1811 |
| **ConvTransformer (Retenu)** | 480k | **51.78%** | **82.08%** | **50.72%** | **1.9043** |

*(Pour rappel, le hasard pur sur 100 classes équivaut à 1.0%).*

## Analyse des comportements

- **Signes les mieux reconnus (> 85%)** : `COMMUNICATION` (92.1%), `INTERPRETE` (90.2%), `RENCONTRER` (88.9%), `C-EST` (86.2%).
  Ce sont des signes bimanités à dynamique spatiale ample et trajectoires distinctes.
- **Signes les plus difficiles (< 20%)** : `APPELER` (13.9%), `DIRE` (14.8%), `PETIT` (16.5%), `ALLER` (17.0%).
  Ce sont des signes très brefs (4 à 8 frames), souvent à une seule main, où les mouvements sont réduits et proches d'autres gestes conversationnels.

## Décision

Le point de contrôle `models/transformer_best.pt` est sélectionné comme modèle par défaut
pour le moteur d'inférence en direct `scripts/live.py`.
Le modèle BiGRU reste disponible en option de secours via `--model-type bigru`.

