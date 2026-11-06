# 003 — Robustesse du téléchargement et filtrage amont

**Date :** 2026-11-06
**Statut :** acceptée

## Contexte

Le dataset LSFB ISOL v2 compte **120 739 instances** réparties sur 4 657 signes.
Pour 3 types de repères (`pose`, `left_hand`, `right_hand`), le téléchargement intégral
représenterait 362 217 requêtes HTTP distinctes.
Lors du test sur échantillon, le téléchargeur officiel `lsfb-dataset` a planté suite à un
timeout HTTP isolé : il n'implémente ni stratégie de retry, ni tolérance aux pannes par fichier.

## Décision

Créer un service de téléchargement dédié dans `lsfb_live.dataset` :
1. **Stratégie de retry robuste** : session `requests` avec `urllib3.Retry` (backoff exponentiel)
   et capture des exceptions par fichier sans faire échouer l'ensemble du lot.
2. **Filtrage par signe (ex: top-100)** : télécharger d'abord les métadonnées (~7 Mo),
   sélectionner les instances cibles, et ne télécharger que leurs poses.
   Pour le top-100, cela représente 55 599 instances (30 850 train, 24 749 test).
3. **Reprise sur erreur** : vérification de l'existence et de l'intégrité minimale des fichiers déjà présents.

## Conséquences

- Pas de gaspillage de bande passante ni de temps sur les 3 500+ signes rares hors du périmètre V1.
- Résilience aux coupures réseau et micro-coupures du serveur de l'UNamur.

