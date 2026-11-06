# Rapport d'Exploration — Dataset LSFB ISOL (Top-100)

**Date :** 2026-11-06  
**Périmètre :** 100 signes les plus fréquents

---

## 1. Distribution des classes

- **Total d'instances :** 55599 (sur les 120 739 du dataset global, soit 46,0%)
- **Signe le plus fréquent :** AUSSI (3308 exemples)
- **Signe le moins fréquent du top :** HABITUDE (217 exemples)
- **Médiane par classe :** 388 exemples
- **Ratio de déséquilibre :** 15.2:1

### Top 5 des signes les plus fréquents
| Signe | Occurrences |
| :--- | :--- |
| AUSSI | 3308 |
| LS | 2984 |
| OUI | 2318 |
| AVOIR | 1546 |
| SOURD | 1443 |

### 5 signes en bas du top 100
| Signe | Occurrences |
| :--- | :--- |
| MOT | 221 |
| COMMENCER | 221 |
| TOUT | 218 |
| BELLE | 218 |
| HABITUDE | 217 |

> [!NOTE]
> Le ratio de déséquilibre (~15:1) impose d'utiliser une **balanced accuracy** et une pondération
> de la perte (Class-Weighted CrossEntropy) lors de l'entraînement.

---

## 2. Répartition par split (indépendance des signeurs)

- **Train :** 30850 instances (59 signeurs distincts)
- **Test officiel :** 24749 instances (40 signeurs distincts)
- **Recouvrement de signeurs :** 0 (garantie de non-fuite d'identité)

---

## 3. Longueur des séquences (en nombre de frames)

- **Minimum :** 4 frames
- **Médiane :** 16 frames (~0.5 seconde à 30 fps)
- **Moyenne :** 19.7 frames
- **90ème percentile :** 32 frames
- **95ème percentile :** 41 frames (~1.3 seconde)
- **99ème percentile :** 70 frames
- **Maximum :** 547 frames
- **Instances vides exclues :** 313

> [!IMPORTANT]
> Une longueur fixe de **40 frames** (interpolation / rééchantillonnage temporel uniforme)
> couvre plus de 95% de la dynamique temporelle des signes sans surcoût mémoire excessif.
