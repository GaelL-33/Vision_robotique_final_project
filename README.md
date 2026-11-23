# Cartographie 3D sémantique

Pilote sur le stockage de descripteurs par instance et par voxel occupé. La feuille de route et le protocole sont dans `ROADMAP.md`. Le premier jalon fournit la lecture du manifeste, la déprojection, la voxelisation géométrique et l'export PLY.

## Démarrer sous PowerShell

Dans le dossier du projet, avec Python 3.13 et un accès aux paquets, créer un environnement isolé.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m openmap smoke
.\.venv\Scripts\python.exe -m openmap inspect runs/smoke/manifest.json --frames 2 --ply runs/smoke/cloud.ply
```

L'environnement préparé sur ce poste réutilise NumPy et Pillow déjà installés via `--system-site-packages`, car le téléchargement de paquets était indisponible. Les deux commandes de géométrie utilisent ce même exécutable. La scène synthétique crée des images PNG 16 bits, deux positions de caméra et deux régions colorées à 2 m. Les points 3D correspondants doivent coïncider. Les fichiers synthétiques et le PLY se trouvent dans `runs/smoke/`.

## Installer et vérifier les modèles

Depuis la racine du dépôt, avec Python 3.12 ou 3.13, Git et un accès réseau pour la première installation. Si ta commande Python s'appelle `python3`, utilise-la dans les exemples. Choisis le même backend pour les deux cartes du benchmark.

```bash
python -m openmap.setup_models --backend cuda
python -m openmap.setup_models --backend cuda --check-only
```

CUDA nécessite un pilote NVIDIA fonctionnel et une distribution de PyTorch compatible avec la plateforme. Utilise `--backend cpu` pour une installation CPU. Le script crée ou réutilise `.venv-cuda/` ou `.venv/` selon le backend, installe les dépendances et place les poids MobileSAM dans `weights/`. Le modèle OpenCLIP du pilote est chargé via Hugging Face dans `cache/huggingface/` par défaut. La vérification exécute un calcul PyTorch, un masque MobileSAM et deux encodages OpenCLIP sur le backend choisi. `--check-only` utilise les modèles déjà installés et le cache Hugging Face hors ligne.

Les options `--venv`, `--weights-dir`, `--hf-home` et `--tmp-dir` permettent de choisir d'autres emplacements. Les chemins du projet peuvent aussi être des liens vers un SSD externe déjà monté. Pour réutiliser ce cache dans une session d'inférence, donner à `HF_HOME` le même chemin avant de lancer Python.

[Versions PyTorch](https://pytorch.org/get-started/previous-versions/) · [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) · [OpenCLIP](https://github.com/mlfoundations/open_clip)

## Choisir et préparer une scène réelle

- **ScanNet v2**, si l'accès est confirmé, prendre une scène annotée et télécharger le `.sens`, le maillage `*_vh_clean_2.ply`, sa segmentation `*.segs.json`, l'agrégation `*.aggregation.json` et le fichier de métadonnées de la scène. Le lecteur ScanNet exporte couleur, profondeur, intrinsèques et poses. Le lecteur Python officiel cible Python 2.7. Pour une machine moderne, préférer un lecteur ScanNet compatible Python 3 déjà validé ou adapter l'exporteur officiel. Utiliser le facteur de profondeur du fichier, `1 / depthShift`, puis recaler le RGB sur le plan profondeur en utilisant les calibrations. Pour les boîtes, regrouper les sommets par instance et appliquer le même éventuel alignement des axes à la géométrie, aux poses et aux annotations. [Données](https://www.scan-net.org/ScanNet/) · [Format du capteur et export](https://github.com/ScanNet/ScanNet/tree/master/SensReader)
- **ARKitScenes**, si ScanNet est toujours en attente lundi à midi, choisir un identifiant depuis `raw/raw_train_val_splits.csv` puis télécharger un seul scan via l'outil du dépôt. Fournir un CSV filtré à un scan à `--video_id_csv` et demander seulement `annotation lowres_depth lowres_wide lowres_wide.traj lowres_wide_intrinsics` avec `--raw_dataset_assets`. Synchroniser les images par timestamp et exploiter les petites images, les intrinsèques et les poses de cette caméra. Partir des utilitaires de pose d'Apple pour convertir la trajectoire et comparer le nuage aux boîtes avant de valider la convention. Garder un deuxième scan pour la mesure si le premier est prêt. [Procédure officielle](https://github.com/apple-aiml-research/ARKitScenes/blob/main/DATA.md)
- **Replica**, si l'équipe possède déjà des séquences rendues RGB-D avec leurs poses et le maillage sémantique correspondant, vérifier sur une scène que les poses et les annotations partagent le même repère. [Structure des scènes Replica](https://github.com/facebookresearch/Replica-Dataset)

Un adaptateur doit écrire un manifeste conforme au format ci-dessous. Contrôler d'abord les contours couleur/profondeur dans chaque trame, puis deux vues éloignées de la même surface dans un visualiseur PLY.

Pour télécharger un seul scan brut ARKitScenes depuis la racine du dépôt, avec `curl` installé et le SSD monté si `data/` y pointe, lancer la commande suivante. Sans arguments, elle prend le scan `47333462` du split `Training`, donné en exemple par Apple.

```bash
python3 -m openmap.download_arkitscenes
```

Les cinq ressources sont enregistrées dans `data/arkitscenes/raw/Training/47333462/`. Pour choisir un autre scan, passer `--video-id` et `--split` avec les valeurs du CSV officiel. Le téléchargement prépare les fichiers bruts. L'adaptateur doit ensuite les synchroniser, vérifier leur recalage et créer le manifeste.

Pour préparer le scan d'exemple et inspecter les dix trames sélectionnées sur la durée du scan :

```bash
python -m openmap.prepare_arkitscenes data/arkitscenes/raw/Training/47333462
python -m openmap inspect runs/47333462/manifest.json --frames 10 --pixel-stride 2 --ply runs/47333462/cloud.ply
```

L'adaptateur associe RGB, profondeur et intrinsèques au même timestamp, puis retient la pose la plus proche dans une tolérance de 2 ms par défaut. Il conserve le repère monde d'Apple. Le contrôle visuel des contours RGB-D et des boîtes annotées reste nécessaire avant le benchmark.

Avec Matplotlib installé via `requirements-analysis.txt`, générer les aperçus et le comptage de points par boîte avec :

```bash
python -m openmap.check_arkitscenes runs/47333462/manifest.json data/arkitscenes/raw/Training/47333462/47333462_3dod_annotation.json --frames 6 7
```

Ouvrir `controle-rgbd.png`, `controle-multivue.png` et `controle-annotations.png` dans `runs/47333462/`. Les numéros des boîtes correspondent aux entrées de `controle.json`, qui consigne aussi les trames, le seuil et le sous-échantillonnage utilisés. Le comptage aide à examiner la couverture des objets.

```powershell
.\.venv\Scripts\python.exe -m openmap inspect data/ma_scene/manifest.json --frames 10 --pixel-stride 2 --voxel-size 0.05 --ply runs/ma_scene/cloud.ply
```

## Contrat du manifeste RGB-D

Fichier JSON UTF-8. Les chemins d'images s'interprètent par rapport au manifeste. L'ordre des entrées est l'ordre temporel. Filtrer les poses manquantes lors de la conversion. Le RGB est déjà aligné sur les pixels profondeur. Une profondeur brute nulle indique une mesure invalide.

```json
{
  "schema_version": 1,
  "scene_id": "ma_scene",
  "depth_scale_to_m": 0.001,
  "frames": [
    {
      "id": "000",
      "rgb": "rgb/000.jpg",
      "depth": "depth/000.png",
      "K": [[580, 0, 320], [0, 580, 240], [0, 0, 1]],
      "T_world_cam": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    }
  ]
}
```

Les coordonnées optiques de la caméra suivent les axes X à droite, Y vers le bas et Z vers l'avant. La profondeur est la distance selon Z en mètres. La pose multiplie un vecteur colonne caméra pour obtenir des coordonnées monde. Les voxels sont indexés par `floor(XYZ / taille_du_voxel)` avec une origine commune.

## Organisation

`openmap/` contient le code Python, `configs/` les paramètres de départ et `skills/` la procédure locale pour les agents. `data/`, `weights/`, `cache/` et `runs/` contiennent respectivement données, poids, observations réutilisables et sorties. Le mode d'emploi de la compétence est référencé dans `AGENTS.md`.

