# 007 — Enrichissement cinématique et architecture Multi-Scale Attention

**Date :** 2026-11-06  
**Statut :** acceptée

## Contexte

La version 1 a démontré la supériorité du ConvTransformer (51,78% Top-1 / 82,08% Top-5)
sur le BiGRU (44,80% Top-1 / 75,76% Top-5) sur le Top-100 des signes LSFB.
Cependant, le modèle n'exploitait que les positions spatiales $(x, y, z)$ brutes, forçant
les couches d'attention à déduire elles-mêmes les vecteurs de mouvement. De plus, la mise
en commun temporelle par simple moyenne (`mean pooling`) accordait le même poids aux phases
de transition qu'au geste pivot.

## Décisions méthodologiques

1. **Vecteurs de vitesse instantanée ($\Delta P$)** :
   - Calcul de la dérivée discrète temporelle $V_t = P_t - P_{t-1}$ sur la séquence normalisée.
   - Concaténation des positions (195) et des vitesses (195) $\to$ **390 caractéristiques par trame**.
   - Traitement strictement symétrique et identique entre entraînement et flux webcam live.

2. **Augmentation par symétrie miroir (Invariance Droitier / Gaucher)** :
   - Inversion de l'axe horizontal $x \leftarrow -x$ à $p=0.5$.
   - Permutation coordonnée des repères symétriques : main gauche $\leftrightarrow$ main droite,
     épaule/coude/poignet gauche $\leftrightarrow$ droit.
   - Doublement virtuel des exemples d'apprentissage pour les signes unimanités.

3. **Architecture Multi-Scale Spatio-Temporal Transformer (MSST-Transformer)** :
   - Convolutions temporelles multi-échelles parallèles ($k = 3, 5, 7$) pour capter
     simultanément les micro-mouvements rapides des doigts et les mouvements amples des bras.
   - Encodeur Transformer avec plongements positionnels apprenables.
   - **Attention Pooling** : mécanisme d'attention scalaire apprenable sur l'axe temporel
     $\sum_t \alpha_t h_t$ permettant d'isoler l'acmé gestuelle du signe.
   - **Label Smoothing (0.1)** pour calibrer les probabilités sur les signes proches.

## Résultats expérimentaux (Top-100, 40 signeurs de test non vus)

| Modèle | Entrées | Top-1 Accuracy | Top-5 Accuracy | Balanced Accuracy | Test Loss |
|---|---|---|---|---|---|
| **v1.0 Baseline (BiGRU)** | 195 (Pos) | 44,80% | 75,76% | 44,66% | 2,1811 |
| **v1.0 ConvTransformer** | 195 (Pos) | 51,78% | 82,08% | 50,72% | 1,9043 |
| **v1.1 MSST-Transformer (Retenu)** | **390 (Pos+Vit)** | **54,46%** | **83,23%** | **54,23%** | **1,7647** |

### Analyse des gains

1. **Top-1 Accuracy (+2,68% vs v1.0, +9,66% vs Baseline)** :
   L'adjonction des vitesses différentielles $\Delta P$ permet au réseau de discriminer
   les trajectoires de mouvement sans ambiguïté géométrique.
2. **Balanced Accuracy (+3,51% vs v1.0, +9,57% vs Baseline)** :
   Le mécanisme d'Attention Pooling et l'invariance miroir droitier/gaucher profitent
   particulièrement aux classes rares du Top-100, réduisant la dispersion des rappels par classe.
3. **Poids sauvegardés** : `models/msst_best.pt` (2,6 Mo).


