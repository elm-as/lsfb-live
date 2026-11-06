# 009 — Avatar paramétrique vectoriel et invariance temporelle au framerate

**Date :** 2026-11-06  
**Statut :** acceptée

## Contexte et retour d'expérience terrain

Lors du déploiement de l'interface interactive pour tester le modèle Top-250 en conditions réelles, j'ai été confronté à un double décalage fondamental entre les données d'entraînement de laboratoire et l'expérience utilisateur face webcam :

1. **L'échec du rejeu des poses brutes pour guider l'utilisateur** :
   Initialement, le mode entraînement rejouait les repères squelettiques extraits directement des vidéos du corpus LSFB ISOL de Namur. Ce choix s'est révélé très décevant à l'usage :
   - Les vidéos sources sont issues de conversations spontanées entre signeurs natifs assis. Les postures sont souvent relâchées, co-articulées d'une seule main (la main passive restant posée sur les cuisses), et déformées par la perspective d'une caméra de studio éloignée.
   - Sans tête proportionnée ni cou anatomique, le squelette ressemblait à un enchevêtrement de bâtons flottants désynchronisés, incapable de transmettre la trajectoire canonique du signe.
   - **L'inversion cognitive du miroir** : la webcam à droite affiche une image miroir (votre main droite est à droite de l'écran), tandis que la vidéo studio à gauche était vue de face (la main droite du signeur était à gauche de l'écran). Ce conflit d'orientation rendait la reproduction motrice contre-intuitive.

2. **Le piège de la fréquence d'échantillonnage (8,9 FPS vs 30 FPS)** :
   L'analyse en direct via MediaPipe Tasks Holistic combinée à l'inférence PyTorch sur un processeur standard d'ordinateur portable tourne à environ **8,9 à 9,5 images par seconde**.
   Dans la première implémentation, la mémoire glissante accumulait 40 trames discrètes. À 8,9 FPS, 40 trames représentaient :
   $$\frac{40 \text{ trames}}{8,9 \text{ FPS}} \approx 4,5 \text{ secondes}$$
   Or, dans le jeu de données d'entraînement, une séquence de 40 trames standardisées correspond à environ **1,3 seconde** de vidéo à 30 FPS.
   **Conséquence directe :** un geste réel d'une seconde exécuté par l'utilisateur ne remplissait que 9 trames dans la mémoire tampon ; les 31 trames restantes étaient composées de mouvements d'hésitation ou de repos antérieurs. Le modèle voyait une séquence étirée d'un facteur 3,5x, diluant complètement la dynamique cinématique.

3. **L'hallucination de signes en phase de repos** :
   En l'absence de classe "neutre / no-sign" dans un classifieur fermé à 250 classes, tout mouvement vers l'avant (comme approcher la main du clavier ou hésiter) générait de fortes probabilités pour des signes directionnels comme `AVANCER` ou `MONTRER`.

---

## Décisions d'ingénierie et d'architecture

### 1. Remplacement du rejeu brut par un Avatar Paramétrique Vectoriel (Base v3)

Plutôt que d'essayer de nettoyer des milliers d'enregistrements bruts bruités, j'ai conçu un module autonome d'avatar vectoriel mathématique ([`src/lsfb_live/display/parametric_avatar.py`](../../src/lsfb_live/display/parametric_avatar.py)) qui constitue la première brique de la **v3 (Text-to-Sign)** :

- **Modèle anatomique proportionné** :
  - Buste trapézoïdal ombré en aplats sobres, clavicules et cou proportionné.
  - Tête expressive avec visage délimité et yeux.
  - Bras articulés avec cinématique inverse simplifiée pour positionner naturellement les coudes en fonction des poignets.
  - Mains stylisées avec configurations canoniques franches : index dressé avec point lumineux pour `RENCONTRER` et `DIFFERENT`, toit oblique pour `MAISON`, mains ouvertes à 5 doigts pour `CORPS` et `AVANCER`.
- **Trajectoires cinématiques mathématiques** :
  Chaque signe canonique est modélisé par une équation paramétrique continue en fonction de l'avancement normalisé $t \in [0, 1]$ avec amorti sinusoïdal (lissage *ease-in-out*) :
  $$s(t) = \frac{1 - \cos(\pi t)}{2}$$
  Ce formalisme mathématique garantit une trajectoire 100% textbook, sans bruit, fluide à 60 FPS et parfaitement calquée sur l'axe miroir de l'utilisateur.

### 2. Invariance temporelle au framerate (Fenêtre stricte de 1,3 seconde)

J'ai refondu le moteur glissant ([`src/lsfb_live/recognition/live_engine.py`](../../src/lsfb_live/recognition/live_engine.py)) pour substituer un comptage de temps réel au comptage de trames :

- Chaque trame entrante est horodatée avec le temps système Unix (`time.time()`).
- La fenêtre glissante extrait systématiquement les trames comprises dans l'intervalle $[t_{\text{actuel}} - 1,3\text{ s}, t_{\text{actuel}}]$.
- Cette sous-séquence (qu'elle contienne 10 trames à 8 FPS ou 40 trames à 30 FPS) est transmise à la fonction `resample_temporal_sequence`, qui l'interpole linéairement vers exactement 40 trames normalisées.
- **Résultat :** la vitesse apparente du geste perçue par le réseau de neurones est rigoureusement identique à celle vue pendant l'entraînement, quel que soit le matériel ou la fluidité de la webcam.

### 3. Filtre d'inactivité et alerte bimanuelle

- **Détection d'état stationnaire** : Si la somme des déplacements absolus des poignets sur la fenêtre est inférieure à un seuil physique ($\Delta P < 0,04$), l'inférence est gelée et l'interface affiche sobrement `En attente d'un geste...`.
- **Vérification d'articulation bimanuelle** : Le catalogue distingue explicitement les signes unimanuels (1 main) et bimanuels (2 mains). Si l'utilisateur tente un signe bimanuel (`DIFFERENT`, `MAISON`, `CORPS`) en ne levant qu'une seule main, un bandeau d'assistance s'affiche : `ATTENTION : 2 MAINS REQUISES (Levez vos deux mains !)`.

---

## Validation et impact

- **Tests unitaires** : La suite passe à 22 tests validés en 4,7 secondes (`pytest tests/`), incluant la validation des bornes cinématiques et de l'intégrité du rendu vectoriel.
- **Résultat utilisateur** : L'interface scindée fournit une démonstration immédiate, symétrique et sans ambiguïté motrice, levant le sentiment de confusion face aux vidéos de studio brutes.
