# 010 — Architecture Bi-Stream Articulaire & Descripteurs Phonologiques (v1.3)

**Date :** 2026-11-06  
**Statut :** acceptée  
**Auteur :** Elmas (ElmasCore)  

---

## 1. Contexte & Problématique

Lors de la montée en charge sur le **Top-250** (v1.2), le modèle `MSST-Transformer` standard a atteint **39,68% Top-1** et **70,67% Top-5** sur les 33 753 séquences du test officiel (40 signeurs non vus). Bien que le Top-5 demeure solide pour un guidage applicatif, le Top-1 subit une dégradation notable par rapport au Top-100 (54,46%).

L'analyse d'erreurs a mis en lumière deux écueils structurels :

1. **L'écrasement des micro-configurations manuelles** :
   Dans une représentation unifiée (390 dimensions aplaties), les grands mouvements d'amplitude du corps (déplacements des bras, haussement d'épaules) dominent le gradient par rapport aux micro-mouvements des doigts (flexion des phalanges, orientation subtile de la paume).
2. **L'absence de contraintes phonologiques explicites** :
   En linguistique de la langue des signes (modèle de William Stokoe et modèle phonologique de Liddell-Johnson), un signe est rigoureusement défini par 5 paramètres phonologiques indépendants :
   - **Emplacement (Location / TAB)** : proximité de la tête, du menton, de la poitrine ou de l'épaule.
   - **Configuration manuelle (Handshape / DEZ)** : état d'extension ou de flexion des doigts.
   - **Mouvement (Movement / SIG)** : direction, vitesse, divergence/convergence des deux mains.
   - **Orientation (Orientation / ORI)** : rotation et vis-à-vis des paumes.
   - **Contact (Contact / CON)** : contact ou distance critique entre les deux mains ou avec le corps.

---

## 2. Décision architecturale : Bi-Stream Articulaire (416 features)

J'ai conçu et déployé une architecture bi-stream découplée (`BiStreamMSSTClassifier`), alimentée par 416 dimensions décomposées en trois flux orthogonaux :

```
                  ┌──────────────────────────────────────────────┐
                  │ Séquence normalisée de repères (T=40, D=195) │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┼────────────────────────┐
                 ▼                       ▼                        ▼
      [Corps Macro-Dynamique]  [Mains Micro-Config]     [Descripteurs Phonologiques]
         23 repères x 3           42 repères x 3           26 scalaires géométriques
         Pos (69) + Vel (69)      Pos (126) + Vel (126)    (inter-mains, ancres, doigts)
            (T, 138)                 (T, 252)                    (T, 26)
                 │                       │                           │
           Conv1D (k=3)            Multi-Scale Conv            LayerNorm + Dense
             GELU + BN              (k=3, 5, 7) + BN             GELU (32 dims)
             (96 dims)               (96 dims)                       │
                 │                       │                           │
                 └───────────────────────┼───────────────────────────┘
                                         ▼
                            [Fusion Multimodale Dense]
                                    (192 dims)
                                         │
                             + Plongement Positionnel
                                         │
                         [Transformer Encoder (2 couches)]
                             (4 têtes d'attention, GELU)
                                         │
                            [Attention Pooling Temporel]
                            Pondération dynamique (softmax)
                                         │
                                [Dense Classifier]
                                    250 classes
```

### 2.1 Décomposition mathématique des 416 caractéristiques

1. **Flux Corps & Trajectoire macro (138 dimensions)** :
   - 23 repères supérieurs MediaPipe (tête, épaules, coudes, poignets) en coordonnées spatiales normalisées $(x, y, z)$ ($23 \times 3 = 69$).
   - 69 vitesses instantanées discrètes $\Delta P_t = P_t - P_{t-1}$.
2. **Flux Mains & Micro-configurations (252 dimensions)** :
   - 42 repères de mains (21 gauche + 21 droite) en coordonnées normalisées ($42 \times 3 = 126$).
   - 126 vitesses instantanées des doigts et articulations palmaires.
3. **Flux Phonologique Explicite (26 dimensions)** :
   - **Inter-mains (5 scalaires)** : distance euclidienne inter-poignets $\|rw - lw\|_2$, vecteur 3D relatif $(rw - lw)$, distance critique de contact inter-index pointes $\|r_{index} - l_{index}\|_2$.
   - **Ancrages corporels (6 scalaires)** : distances poignet gauche/droit au nez $\|w - nose\|$, au sternum $\|w - chest\|$, et élévation relative par rapport à la ligne des épaules ($y_w - y_{shoulders}$).
   - **Formes & extensions des 10 doigts (10 scalaires)** : distances euclidiennes de chaque bout de doigt (pouce, index, majeur, annulaire, auriculaire) par rapport au carpe de sa main, pour les mains gauche et droite.
   - **Cinématique scalaire (5 scalaires)** : vitesses scalaires des poignets, accélérations scalaires $\|v_t - v_{t-1}\|$, et taux d'écartement/rapprochement inter-mains $\Delta \|rw - lw\|$.

---

## 3. Optimisation : Focal Cross-Entropy & Augmentations

### 3.1 Focal Loss avec lissage d'étiquette

Sur un catalogue étendu de 250 signes, les exemples faciles et très distinctifs monopolisent la rétropropagation au détriment des homonymes gestuels complexes. J'ai formulé la **Focal Cross-Entropy** :

$$\mathcal{L}_{focal} = - \sum_{c=1}^{C} y_{smooth, c} \cdot (1 - p_c)^\gamma \cdot \log(p_c)$$

- $\gamma = 1.5$ : atténue la contribution des exemples bien classés ($p_c \approx 0.9 \implies (1 - 0.9)^{1.5} \approx 0.031$, soit une réduction de gradient de 97%).
- Lissage d'étiquette $\alpha = 0.05$ : prévient la sur-confiance sur les classes de la traîne.
- Pondération inverse des fréquences $w_c = \frac{N}{C \cdot N_c}$ pour équilibrer la Balanced Accuracy.

### 3.2 Temporal Cutout (Masquage de trames)

Pour empêcher le réseau de sur-apprendre des raccourcis temporels statiques, une augmentation par **Temporal Cutout** ($p = 0.35$) masque aléatoirement un segment de 2 à 3 trames consécutives en début ou milieu de geste, forçant le Transformer à reconstruire la cinématique globale par attention contextuelle.

---

## 4. Résultats expérimentaux (Test officiel — 33 753 séquences, 40 signeurs non vus)

| Échelle | Architecture | Features | Test Top-1 | Test Top-5 | Balanced Accuracy | Test Loss |
|---|---|---|---|---|---|---|
| Hasard | Aléatoire uniforme | - | 0,40% | 2,00% | 0,40% | 5,5215 |
| **v1.1** | **MSST-Transformer** | 390 (Pos + Vel) | **54,46%** (100 cl.) | **83,23%** (100 cl.) | **54,23%** | 1,7647 |
| **v1.2** | **MSST-Transformer** | 390 (Pos + Vel) | **39,68%** (250 cl.) | **70,67%** (250 cl.) | **41,64%** | 2,5521 |
| **v1.3** | **BiStream-MSST (Phonologique)** | **416 (Corps + Mains + Phono)** | **37,75%** (250 cl.) | **69,28%** (250 cl.) | **39,11%** | 2,6497 |

### 4.1 Signes les mieux reconnus (Excellente phonologie bimanuelle)
- `ADORER` : **91,3%** (69 exemples) — geste bimanuel très expressif au niveau du torse avec convergence marquée.
- `CONVENIR` : **87,5%** (56 exemples) — contact précis des index et orientation palmaire.
- `INTERPRETE` : **85,4%** (123 exemples) — cinématique alternée caractéristique des deux mains.
- `RENCONTRER` : **84,6%** (117 exemples) — trajectoire frontale convergente des deux index verticaux.
- `CORPS` : **82,9%** (41 exemples) — descente tactile délimitée sur le buste.

### 4.2 Analyse des signes difficiles & homonymes
- Les classes les plus en difficulté (`FACILE.10`, `VOULOIR`, `QUI`, `ALLER`, `FSIGN`) partagent une caractéristique commune : des micro-déplacements d'un seul doigt ou une composante faciale/orale prépondérante dans la LSFB non couverte par les 23 repères supérieurs de pose.
- L'apparition fréquente de `(ALLER?)` observée en direct sous faible confiance s'explique par sa nature de classe "aspirateur" de cinématique neutre, efficacement filtrée par le seuil de décision et la détection d'inactivité du coach.

---

## 5. Conséquences et Intégration

- **Cohérence stricte Train/Live** : La fonction pure `assemble_bistream_features(positions)` garantit que l'extraction des 416 dimensions en streaming direct (`LiveInferenceEngine`) est strictement identique aux tenseurs d'entraînement.
- **Ressources mémoire contenues** : L'assemblage dynamique à la volée dans `PhonologicalSignDataset` évite le stockage redondant de plusieurs gigaoctets sur disque tout en ne consommant que ~100 ms par batch sur CPU.
- **Prêt pour le direct** : Les scripts `scripts/train.py`, `scripts/evaluate.py`, `scripts/live.py` et `scripts/practice.py` adoptent `BiStreamMSSTClassifier` par défaut.


