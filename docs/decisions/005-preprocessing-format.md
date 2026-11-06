# 005 — Stockage et découpage des données prétraitées

**Date :** 2026-11-06  
**Statut :** acceptée

## Contexte

Lire 55 000 fichiers `.npy` individuels pendant la boucle d'entraînement saturerait les I/O
disque et ralentirait l'apprentissage CPU par un facteur 10.

## Décision

Compiler les tenseurs prétraités normalisés en fichiers NumPy consolidés :
- `data/processed/train_features.npy` (formes: $N_{train} \times 40 \times 195$, float32 ou float16)
- `data/processed/train_labels.npy` ($N_{train}$, int64)
- `data/processed/val_features.npy`
- `data/processed/val_labels.npy`
- `data/processed/test_features.npy`
- `data/processed/test_labels.npy`
- `data/processed/class_mapping.json` (dictionnaire bi-directionnel index $\leftrightarrow$ glose)
- `data/processed/class_weights.npy` (poids inverses des fréquences pour la loss)

## Découpage des splits (indépendance des signeurs)

- **Train** : Folds 2 et 3 (~39 signeurs, ~21 000 instances)
- **Validation** : Fold 4 (20 signeurs, ~9 500 instances)
- **Test** : Folds 0 et 1 (40 signeurs, ~24 700 instances)
Aucun signeur n'est partagé entre train, val et test.

