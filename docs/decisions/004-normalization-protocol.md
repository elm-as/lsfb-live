# 004 — Protocole de normalisation des repères

**Date :** 2026-11-06  
**Statut :** acceptée

## Contexte

Les données d'apprentissage proviennent de signeurs filmés en studio à distance variable,
tandis que l'inférence live sera réalisée face à une webcam de portable à cadrage proche.
Sans normalisation rigoureuse et invariante, le modèle échouerait en production.

## Décision

Le pipeline de normalisation standardisé dans `lsfb_live.landmarks` applique :

1. **Sélection des repères (65 points)** :
   - 23 repères de torse/tête (indices MediaPipe Pose 0 à 22 : tête, épaules, coudes, poignets)
   - 21 repères de main gauche
   - 21 repères de main droite
   *(Les jambes 23-32 sont ignorées car absentes du champ de vision et sources de bruit).*

2. **Centrage spatial (invariance à la position dans l'image)** :
   - Origine = point médian entre l'épaule gauche (index 11) et l'épaule droite (index 12).
   - Centrage appliqué à chaque frame pour éliminer tout déplacement du buste.

3. **Mise à l'échelle (invariance à la distance à la caméra)** :
   - Facteur d'échelle = distance euclidienne inter-épaules $\|P_{11} - P_{12}\|_2$.
   - Division des coordonnées par cette distance (avec $\epsilon = 10^{-6}$).

4. **Échantillonnage temporel uniforme ($T = 40$ frames)** :
   - Interpolation linéaire sur l'axe temporel vers une taille fixe de 40 frames.
   - 40 frames couvrent 95% des signes du corpus (médiane à 16 frames).

5. **Gestion des mains absentes** :
   - Remplacement systématique des `NaN` par 0.0.

## Règle absolue

Ce module est pur, déterministe et partagé sans aucune duplication entre le prétraitement
du dataset d'entraînement et la boucle de capture webcam en temps réel.

