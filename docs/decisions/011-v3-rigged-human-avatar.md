# ADR 011 — Architecture V3 : Mannequin Humain Rigué 3D, Cinématique Inverse (IK) & Studio Text-to-Sign

**Statut :** Validé et Déployé (V3.0)  
**Date :** Novembre 2026  

---

## 1. Contexte et Problématique

Dans les versions antérieures (v1.3 / v1.4), l'avatar du mode Coach reposait sur une esquisse vectorielle 2D simplifiée (`parametric_avatar.py`), où chaque main était réduite à quelques traits stylisés en 2D. Cette représentation ne permettait pas :
1. De restituer fidèlement la **configuration 3D réelle des 21 articulations de chaque main** (orientation palmaire, flexion inter-phalangienne, opposition du pouce).
2. De tourner la caméra en 3D autour du signeur (vue de face, profil 30°, vue oblique) pour observer la profondeur $Z$ du geste.
3. D'animer l'avatar sur l'ensemble des **30 signes du modèle v1.5** à partir des trajectoires 3D réelles des signeurs natifs du corpus LSFB.
4. De piloter ce même mannequin 3D en **Motion Capture temps réel (Retargeting)** depuis la webcam ou via un **Studio Web 3D (Three.js)** pour le Text-to-Sign.

---

## 2. Décision Architecturale : Domaine `src/lsfb_live/avatar/`

Conformément à la charte **ElmasCore**, j'ai créé le domaine métier dédié [`src/lsfb_live/avatar/`](../../src/lsfb_live/avatar/) séparant strictement la géométrie cinématique 3D de la couche d'affichage :

1. **Extraction des Prototypes 3D Natifs (`models/sign_prototypes_3d.json`) :**
   - Pour chacun des 30 signes lexicaux, sélection automatique de l'instance médoïde (trajectoire la plus proche de la médiane de classe présentant le moindre jerk cinématique et une visibilité complète des mains).
   - Lissage temporel gaussien sans déphasage et sous-échantillonnage à 24 trames clés 3D `(24, 65, 3)`.

2. **Squelette Rigué 3D & Cinématique Inverse Analytique ([`rig_skeleton.py`](../../src/lsfb_live/avatar/rig_skeleton.py)) :**
   - `RigSkeleton3D` : structure immuable décrivant la tête orientée, le cou, le bassin, les épaules, les coudes, les poignets et les $2 \times 21 = 42$ articulations phalangiennes 3D en espace métrique normalisé par la largeur bi-acromiale.
   - `solve_two_bone_arm_ik` : résolution analytique exacte de la **Cinématique Inverse (IK à 2 os)** dans le triangle $(\text{Épaule}, \text{Coude}, \text{Poignet})$ via la loi des cosinus et l'orthogonalisation de Gram-Schmidt le long d'un vecteur pôle anatomique ($\vec{p}_{\text{pole}}$). Les longueurs d'humérus ($L_1 = 0.68$) et de radius/ulna ($L_2 = 0.64$) demeurent strictement constantes.
   - `_anchor_and_scale_hand` : ancre les 21 phalanges 3D sur chaque poignet et normalise la taille de la paume à une proportion anatomique constante ($0.25$), avec génération automatique d'une main 3D au repos (`build_relaxed_hand_3d`) lorsque la main non dominante est inactive.

3. **Séquenceur Text-to-Sign & Lemmatisation ([`text_to_sign_sequencer.py`](../../src/lsfb_live/avatar/text_to_sign_sequencer.py)) :**
   - `parse_french_to_lsfb_glosses` : convertit une phrase en français naturel (ex: *"Mon papa aime toujours la famille et explique l'histoire à l'école"*) en séquence de gloses LSFB (`["PAPA", "AIMER", "TOUJOURS", "FAMILLE", "EXPLIQUER", "HISTOIRE", "ECOLE"]`).
   - `SignSequenceTimeline` : enchaîne les prototypes 3D avec une fenêtre de transition de 20% interpolée par lissage cosinus et recalcul IK continu (`interpolate_rig_skeletons`).

4. **Double Moteur de Rendu 3D (Hybride OpenCV + Three.js WebGL) :**
   - **OpenCV Volumétrique ([`rig_renderer_cv.py`](../../src/lsfb_live/display/rig_renderer_cv.py)) :** Projection perspective avec rotation orbitale (`yaw_deg`), tri de profondeur $Z$ (algorithme du peintre), capsules cylindriques ombrées à 3 niveaux (ombre, matériau, reflet spéculaire) et paumes polygonales 3D.
   - **Studio Web 3D Three.js ([`scripts/studio_v3.py`](../../scripts/studio_v3.py) & [`index.html`](../../src/lsfb_live/display/web_studio/index.html)) :** Visionneuse WebGL 3D interactive avec éclairage 3 points, caméra `OrbitControls` 360°, saisie de phrases Text-to-Sign, catalogue des 30 signes 3D et mode Motion Capture Live.

