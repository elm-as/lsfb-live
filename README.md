# LSFB Live — Reconnaissance en Direct de la Langue des Signes (LSFB)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%20(CPU)-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-1.2.2-0072B2?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Vocabulary](https://img.shields.io/badge/Vocabulary-250%20Signs%20(v1.2)-darkblue?style=for-the-badge)](#4-résultats-expérimentaux--évaluation-sur-signeurs-inconnus)
[![Top-1 Accuracy](https://img.shields.io/badge/Top--1%20(250%20Signs)-39.68%25-brightgreen?style=for-the-badge)](#4-résultats-expérimentaux--évaluation-sur-signeurs-inconnus)
[![Top-5 Accuracy](https://img.shields.io/badge/Top--5%20(250%20Signs)-70.67%25-success?style=for-the-badge)](#4-résultats-expérimentaux--évaluation-sur-signeurs-inconnus)
[![Signer-Independent](https://img.shields.io/badge/Evaluation-40%20Unseen%20Signers-blueviolet?style=for-the-badge)](#4-résultats-expérimentaux--évaluation-sur-signeurs-inconnus)

*Un pipeline complet de Data Science : du corpus de poses de l'Université de Namur jusqu'à la classification temps réel face webcam sur un processeur standard d'ordinateur portable, sans carte graphique dédiée.*

</div>

---

## 1. Contexte & Présentation du Dataset LSFB

Dans le cadre de mes études en **Data Science**, j'ai voulu construire un projet centré sur la réalité de la modélisation statistique et séquentielle : gestion de distributions asymétriques, normalisation spatio-temporelle invariante et inférence temps réel sous contraintes de calcul.

### Le Corpus LSFB (Université de Namur)
La **LSFB** (Langue des Signes de Belgique Francophone) est la langue visuelle employée par la communauté sourde en Wallonie et à Bruxelles.

Le dataset utilisé est issu des travaux de recherche de l'**Université de Namur (UNamur)** :
- **Source :** Le *Corpus LSFB*, recensant des dizaines d'heures d'échanges spontanés entre une centaine de signeurs natifs, annotés par des linguistes.
- **Sous-ensemble LSFB-ISOL :** Une collection de clips segmentés signe par signe (120 739 instances réparties sur 4 657 classes lexicales).
- **Format Poses (Landmarks) :** Plutôt que de manipuler des flux vidéo lourds, le projet s'appuie directement sur les squelettes 3D nettoyés (coordonnées des articulations du corps et des mains extraites via MediaPipe). Ce choix assure une **cohérence parfaite** : le modèle est entraîné sur le format géométrique exact que la webcam produit en direct.

---

## 2. L'Exploration et les Caractéristiques des Données

L'analyse exploratoire complète menée sur le corpus est consignée dans [`docs/exploration_report.md`](docs/exploration_report.md).

```
   120 739 instances (LSFB ISOL global, 4 657 classes)
               │
               ├──► Top-100 signes (55 286 instances validées, v1.0 & v1.1)
               │      ├── Train : 21 401 instances (39 signeurs)
               │      ├── Val   :  9 136 instances (20 signeurs)
               │      └── Test  : 24 749 instances (40 signeurs non vus)
               │
               └──► Top-250 signes (75 825 instances validées, v1.2)
                      ├── Train : 29 324 instances (39 signeurs)
                      ├── Val   : 12 748 instances (20 signeurs)
                      └── Test  : 33 753 instances (40 signeurs non vus)
```

### Le déséquilibre de classes (15.2:1)
Même restreint au top 100, la distribution présente une forte asymétrie :
- Le signe le plus fréquent (`AUSSI`) compte **3 308 exemples**.
- Le 100ème signe (`HABITUDE`) n'en compte que **217**.
- Médiane par classe : 388 exemples.

> **Impact méthodologique :** Une exactitude brute (*Accuracy Top-1*) serait trompeuse face à un tel déséquilibre. L'évaluation repose donc sur la **Balanced Accuracy** (moyenne des rappels par classe), et l'entraînement intègre une fonction de perte **Class-Weighted CrossEntropy** proportionnelle à l'inverse des fréquences de chaque signe.

### Analyse temporelle et fenêtrage
L'étude de la longueur des séquences a mis en évidence :
- **Longueur minimale :** 4 frames.
- **Longueur médiane :** **16 frames** (~0,53 seconde à 30 fps).
- **Moyenne :** 19,7 frames.
- **95ème percentile :** **41 frames** (~1,36 seconde).

> **Choix de modélisation :** Une séquence standardisée à **$T = 40$ frames** par rééchantillonnage linéaire permet de capter l'intégralité de la dynamique de plus de 95% des signes, tout en maintenant une empreinte mémoire compacte.

---

## 3. Pipeline de Normalisation Invariante

Pour qu'un modèle entraîné sur des enregistrements de studio fonctionne face à une webcam personnelle (cadrage rapproché, utilisateur décentré), les données subissent une normalisation géométrique stricte.

Ce pipeline est centralisé et partagé sans duplication dans [`src/lsfb_live/landmarks/`](src/lsfb_live/landmarks/) :

```mermaid
flowchart LR
    subgraph Capture ["Repères MediaPipe (F frames)"]
        P["Pose (33 pts)"]
        LH["Main Gauche (21 pts)"]
        RH["Main Droite (21 pts)"]
    end

    subgraph Selection ["Sélection Géométrique"]
        UP["23 repères Torse/Tête"]
        UP --- LH
        UP --- RH
    end

    subgraph Invariance ["Invariance Spatiale"]
        C["Centrage sur milieu épaules (P11 + P12)/2"]
        S["Échelle par largeur inter-épaules ||P11 - P12||"]
    end

    subgraph Temporel ["Standardisation Temporelle"]
        T["Interpolation Linéaire vers T = 40 frames"]
    end

    subgraph Sortie ["Tenseur Modèle"]
        OUT["Tenseur 40 x 195 (float32)"]
    end

    Capture --> Selection --> Invariance --> Temporel --> Sortie
```

1. **Sélection de 65 repères anatomiques :**
   - 23 repères supérieurs MediaPipe Pose (nez, yeux, bouche, épaules, coudes, poignets). Les repères des membres inférieurs (23 à 32) sont écartés car hors champ et générateurs de bruit.
   - 21 repères pour chaque main.
   - Total : $65 \times 3 = 195$ caractéristiques par trame.
2. **Invariance de translation :** Centrage systématique de chaque trame sur le milieu des épaules.
3. **Invariance de distance / échelle :** Normalisation par la distance euclidienne inter-épaules $\|P_{11} - P_{12}\|_2$. Que l'utilisateur soit près ou loin de l'écran, l'échelle du squelette demeure identique.
4. **Occultations :** Les mains non détectées (`NaN`) sont substituées par 0.0 de façon déterministe.

---

## 4. Résultats Expérimentaux : Évaluation sur Signeurs Inconnus

Afin de mesurer la capacité réelle de généralisation, le protocole respecte une séparation stricte : **les 40 signeurs du split de test n'ont jamais été vus pendant l'entraînement.**

Trois étapes d'optimisation séquentielle sur CPU ont été comparées :

### Évolution des Modèles sur le Top-100 (v1.0 & v1.1)

| Version | Modèle | Caractéristiques | Top-1 Accuracy | Top-5 Accuracy | Balanced Accuracy | Test Loss |
|---|---|---|---|---|---|---|
| **v1.0** | Baseline BiGRU | 195 (Positions) | 44,80% | 75,76% | 44,66% | 2,1811 |
| **v1.0** | ConvTransformer | 195 (Positions) | 51,78% | 82,08% | 50,72% | 1,9043 |
| **v1.1** | **MSST-Transformer** | **390 (Pos + Vitesses $\Delta P$)** | **54,46%** | **83,23%** | **54,23%** | **1,7647** |

*(Le niveau du hasard sur 100 classes est de 1,0%).*

### Montée en Charge sur le Top-250 (v1.2)

Le passage de 100 à 250 classes multiplie la combinatoire par 2,5 et densifie fortement l'espace des homonymes gestuels. Évalué sur **33 753 séquences de test** issues de 40 signeurs non vus :

| Échelle | Modèle | Classes | Test Top-1 | Test Top-5 | Balanced Accuracy | Test Loss |
|---|---|---|---|---|---|---|
| *Hasard* | *Aléatoire uniforme* | *250* | *0,40%* | *2,00%* | *0,40%* | *5,5215* |
| **v1.1** | MSST-Transformer | 100 | 54,46% | 83,23% | 54,23% | 1,7647 |
| **v1.2** | **MSST-Transformer** | **250** | **39,68%** | **70,67%** | **41,64%** | **2,5521** |

> **Constat majeur :** Malgré l'explosion du nombre de classes, le modèle maintient **plus de 70% en Top-5** et une **Balanced Accuracy supérieure au Top-1 brut (41,64% vs 39,68%)**. La pondération inverse de fréquence protège efficacement les classes rares contre l'écrasement par les classes dominantes.

### Apports méthodologiques clés (MSST-Transformer) :
1. **Vitesses articulaires instantanées ($\Delta P$) :** La dérivation temporelle discrète double l'information cinématique (390 features/frame), levant les ambiguïtés sur la dynamique des trajectoires.
2. **Invariance miroir droitier / gaucher :** Permutation symétrique des membres et inversion $x \leftarrow -x$ à $p=0.5$, doublant la représentativité des gestes asymétriques.
3. **Multi-Scale Convolutions ($k=3, 5, 7$) :** Capture simultanée des micro-mouvements des doigts et des mouvements amples des bras.
4. **Attention Pooling apprenable :** Pondère les trames discriminantes du geste plutôt que de noyer l'information dans une moyenne uniforme.

### Analyse qualitative par signe (Top-250, v1.2)

```
Signes les mieux reconnus (> 85%) :
  1. ADORER         [======================================  ] 92.8%  (69 exemples)
  2. CORPS          [=====================================   ] 90.2%  (41 exemples)
  3. INTERPRETE     [====================================    ] 88.6%  (123 exemples)
  4. RENCONTRER     [====================================    ] 88.0%  (117 exemples)
  5. CONVENIR       [===================================     ] 87.5%  (56 exemples)

Signes les plus délicats (< 5%) :
  1. FSIGN (Dactylologie) [=                                 ]  0.9%  (108 exemples)
  2. ALLER                [=                                 ]  2.7%  (182 exemples)
  3. AUTREFOIS            [=                                 ]  3.3%  (61 exemples)
  4. SAVOIR-NEG           [==                                ]  3.7%  (54 exemples)
  5. VOULOIR              [==                                ]  4.7%  (64 exemples)
```

---

## 5. Architecture du Projet

Le projet applique les principes d'architecture logicielle par domaine de la charte **ElmasCore** (séparation stricte, fonctions pures, 350 lignes max par fichier) :

```
src/lsfb_live/
├── dataset/         Téléchargement résilient, indexation et compilation des tenseurs
├── landmarks/       Constantes MediaPipe et pipeline de normalisation invariante (+ vitesses)
├── recognition/     Modèles (BiGRU, ConvTransformer, MSST), entraînement pondéré, moteur live
├── capture/         Gestion du flux webcam et extracteur MediaPipe Tasks Holistic
└── display/         Rendu OpenCV du squelette articulé et bandeau de prédiction
```

La suite de 20 tests unitaires valide l'intégrité de l'ensemble de la chaîne :
```powershell
pytest -v
# ================= 20 passed in 7.27s =================
```

---

## 6. Guide de Démarrage Rapide

Les poids du modèle retenu v1.2 (`models/msst_best.pt`, 2,68 Mo) entraîné sur le **Top-250** ainsi que le dictionnaire des 250 classes (`models/class_mapping.json`) sont directement inclus dans le dépôt (les poids v1.1 sur 100 classes restent disponibles dans `models/msst_top100.pt`).

### 1. Installation
```powershell
git clone https://github.com/elm-as/lsfb-live.git
cd lsfb-live

# Environnement Python 3.12 (requis pour MediaPipe Tasks)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Lancer la reconnaissance en direct (Webcam)
```powershell
python scripts/live.py
```
*Le modèle MediaPipe Holistic (`holistic_landmarker.task`, 13 Mo) est téléchargé automatiquement dès le premier lancement si absent.*

- **Contrôles :**
  - Placez-vous face caméra (buste et mains visibles).
  - Le squelette articulé s'affiche en surimpression temps réel.
  - Le signe détecté et le niveau de confiance s'affichent en haut à gauche.
  - Le top-3 des candidats alternatifs est affiché en bas à gauche.
  - Appuyez sur **`Q`** ou **`Échap`** pour quitter.

### 3. Entraîner et évaluer un modèle (Optionnel)
```powershell
# Entraîner le MSST-Transformer (v1.1)
python scripts/train.py --model msst --epochs 12

# Évaluer sur les 40 signeurs du split de test
python scripts/evaluate.py --model-path models/msst_best.pt --model-type msst --use-velocity
```

---

## 7. Roadmap & Perspectives

- **v1.0 (Validé) :** Pipeline de base Top-100 (195 features, ConvTransformer à 51,8% Top-1 / 82,1% Top-5).
- **v1.1 (Validé) :** Enrichissement cinématique (390 features positions + vitesses, symétrie miroir, MSST-Transformer à **54,5% Top-1 / 83,2% Top-5 / 54,2% Balanced**).
- **v1.2 (Validé) :** Montée en charge sur le **Top-250** des signes LSFB (75 825 instances réelles, 33 753 séquences de test sur 40 signeurs non vus, **70,7% Top-5** et **41,6% Balanced Acc**).
- **v2.0 (Prochaine étape) :** Reconnaissance continue sur flux non segmenté (LSFB-CONT) couplée à un modèle de langue (LLM) pour la traduction en phrases complètes.
- **v3.0 (Génération inverse) :** Text-to-Sign : synthèse de coordonnées 3D de squelettes pour animer un avatar gestuel.

---

## Références

- **Corpus LSFB (Université de Namur) :** J. Fink, T. Hoyoux, R. Vander Donckt. *The LSFB Dataset: A Corpus for French Belgian Sign Language Recognition*.
- **Google Research :** Framework *MediaPipe* et bibliothèque *Tasks Vision*.

*Projet universitaire en Data Science — 2026.*
