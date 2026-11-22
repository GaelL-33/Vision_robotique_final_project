# Projet_final

Construire une carte 3D sémantique open-vocabulary et mesurer le compromis entre la fusion d'instances et une carte à descripteur par voxel occupé. La présentation est le vendredi 27 novembre 2026. Lire `ROADMAP.md` pour connaître le périmètre et `README.md` pour les commandes et le contrat RGB-D. Consulter la compétence locale `skills/cartographie-rgbd/SKILL.md` pour une tâche sur le pipeline ou les mesures.

## Travailler à plusieurs agents

- Définir l'interface et les fichiers pris en charge avant de paralléliser. Donner à un seul agent la responsabilité de chaque fichier partagé. Planifier des opérations GPU séquentielles pour les mesures.
- Livrer une tranche exécutable depuis une entrée normale, avec la commande exécutée, le résultat observé et le prochain blocage. Tester les changements de géométrie avec `python -m openmap smoke`.
- Utiliser des tableaux XYZ en mètres et une transformation homogène caméra vers monde sur des vecteurs colonnes. Aligner le RGB dans le repère de la profondeur avant de créer le manifeste. Appliquer les changements de repère aux données et aux annotations au même endroit.
- Utiliser la configuration de référence dans `configs/pilot.json`. Consigner les paramètres réellement utilisés et les mêler aux résultats seulement au moment de l'analyse.
- Utiliser `data/` pour les scènes, `weights/` pour les poids, `cache/` pour les observations réutilisables et `runs/` pour les sorties. Conserver les autorisations et archives volumineuses hors du code. Toute métrique de présentation doit être traçable aux sorties d'une exécution sur une scène réelle.

