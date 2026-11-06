# 008 — Montée en charge sur le Top-250 des signes LSFB (v1.2)

**Date :** 2026-11-06  
**Statut :** acceptée

## Contexte et objectifs

Après avoir validé l'architecture **MSST-Transformer** (Multi-Scale Spatio-Temporal Transformer avec enrichissement cinématique et Attention Pooling) atteignant **54,46% Top-1** et **83,23% Top-5** sur 100 classes, l'étape suivante consiste à élargir considérablement le vocabulaire exploitable en passant à **250 signes** (Top-250, version v1.2).

Cette montée en charge constitue un défi majeur en reconnaissance gestuelle :
1. **Multiplication par 2,5 du nombre de classes** : l'espace de décision devient beaucoup plus dense avec de nombreux quasi-homonymes gestuels (signes partageant la même configuration de main ou le même lieu d'articulation).
2. **Accentuation du déséquilibre de classe** : les classes du rang 100 à 250 ont une volumétrie plus restreinte dans le corpus LSFB ISOL, accentuant la longue traîne.
3. **Maintien de la généralisation sur signeurs non vus** : le split de test officiel conserve 40 signeurs jamais rencontrés lors de l'entraînement.

## Volumétrie des données Top-250

- **Instances totales prétraitées** : 75 825 instances réparties sur 250 classes (314 instances vides ou sans repères valides écartées).
- **Split d'entraînement (folds 2 et 3)** : 29 324 séquences (39 signeurs).
- **Split de validation (fold 4)** : 12 748 séquences (20 signeurs).
- **Split de test officiel (folds 0 et 1)** : 33 753 séquences (**40 signeurs non vus**).
- **Dimensions d'entrée** : $(T=40, D=390)$ combinant coordonnées spatiales normalisées (195) et vitesses instantanées ($\Delta P$, 195).

## Adaptations méthodologiques

1. **Pondération inverse des fréquences pour 250 classes** :
   Calcul des poids de classe $w_c = \frac{N}{C \cdot N_c}$ appliqué à la fonction de perte Cross-Entropy avec lissage d'étiquette (`label_smoothing=0.05`), garantissant que les signes rares reçoivent un gradient proportionnel sans être écrasés par les signes ultra-fréquents comme `AUSSI` ou `VOIR`.
2. **Attention Pooling stabilisé** :
   Le mécanisme d'attention temporelle pondère les 40 pas de temps pour chaque séquence, isolant l'apogée du mouvement gestuel sans diluer l'information.
3. **Invariance miroir et régularisation** :
   Augmentation dynamique en cours d'entraînement par symétrie horizontale $x \leftarrow -x$ et permutation des repères bilatéraux pour doubler virtuellement les exemples sur les classes de la traîne.

## Résultats expérimentaux (Test officiel — 40 signeurs non vus)

| Échelle | Modèle | Classes | Test Top-1 | Test Top-5 | Balanced Accuracy | Test Loss |
|---|---|---|---|---|---|---|
| Hasard | Aléatoire uniforme | 250 | 0,40% | 2,00% | 0,40% | 5,5215 |
| **v1.1** | **MSST-Transformer** | **100** | **54,46%** | **83,23%** | **54,23%** | **1,7647** |
| **v1.2** | **MSST-Transformer** | **250** | **39,68%** | **70,67%** | **41,64%** | **2,5521** |

### Analyse des performances

1. **Robustesse du Top-5 (> 70%)** :
   Malgré une explosion de la combinatoire ($\times 2,5$ classes), dans plus de 7 cas sur 10, le bon signe figure dans le Top-5 proposé par le modèle. Dans une application interactive avec suggestion de complétion ou HUD en direct, cela assure une excellente utilisabilité.
2. **Balanced Accuracy supérieure au Top-1 brut (41,64% vs 39,68%)** :
   La pondération inverse de fréquence a parfaitement fonctionné : le modèle ne s'est pas réfugié dans la sur-prédiction des classes majoritaires, maintenant un taux de rappel moyen élevé sur l'ensemble des 250 classes.
3. **Top 5 des signes les mieux reconnus** :
   - `ADORER` : 92,8% (69 exemples) — geste bimanuel très expressif au niveau du torse avec mouvement convergent marqué.
   - `CORPS` : 90,2% (41 exemples) — contact tactile délimité sur le buste.
   - `INTERPRETE` : 88,6% (123 exemples) — cinématique caractéristique des mains en rotation alternée.
   - `RENCONTRER` : 88,0% (117 exemples) — trajectoire frontale convergente des deux index.
   - `CONVENIR` : 87,5% (56 exemples) — trajectoire univoque descendante.
4. **Analyse des signes difficiles** :
   - `FSIGN` (0,9%) : correspond à la dactylologie (épellation manuelle rapide). Sans vidéo haute résolution des micro-phalanges, les squelettes MediaPipe 33+21+21 lissent excessivement les configurations fines.
   - `ALLER` (2,7%), `AUTREFOIS` (3,3%), `SAVOIR-NEG` (3,7%) : signes très courts souvent fondus avec les co-articulations ou confondus avec des particules directionnelles.


