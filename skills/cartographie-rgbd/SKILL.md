---
name: cartographie-rgbd
description: "Mettre en œuvre, vérifier ou mesurer la cartographie 3D open-vocabulary du projet. Utiliser pour l'adaptation RGB-D, la déprojection, MobileSAM et CLIP, la fusion d'instances ou par voxel, les requêtes textuelles et l'évaluation comparative."
---

# Cartographie sémantique RGB-D

1. Lire `AGENTS.md`, puis le jalon pertinent de `ROADMAP.md`. Repérer l'entrée exécutable et les sorties de la dernière expérience avant de coder.
2. Pour une nouvelle scène, produire le manifeste décrit dans `README.md`. Inspecter deux vues éloignées de la même surface. Contrôler la profondeur métrique, l'alignement RGB/D, le sens de `T_world_cam`, l'ordre des axes et le repère des annotations.
3. Exécuter `python -m openmap smoke` avant de modifier une étape en aval. Passer ensuite à quelques trames réelles et garder les paramètres qui permettent de relancer l'essai.
4. Mettre en cache les masques, leur confiance, leurs dimensions et les vecteurs CLIP normalisés pour les deux branches. Inclure dans la clé la scène, les indices exacts des trames et les versions et paramètres des modèles. Garder la profondeur et les poses dans le lecteur commun.
5. Pour comparer les deux cartes, reprendre les mêmes trames, masques, descripteurs, prompts, pas spatial, budget de propositions, précision numérique et matériel. Garder les coûts partagés visibles dans les mesures de bout en bout.
6. Pour chaque exécution, consigner la commande, le dataset, les paramètres, le backend, les effectifs, les erreurs et les chemins des sorties. Suivre le protocole de `ROADMAP.md` pour les métriques et le découpage pilote/évaluation.

