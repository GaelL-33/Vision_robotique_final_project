"""Télécharger les cinq ressources RGB-D d'un scan ARKitScenes."""

import argparse
from pathlib import Path
import subprocess
import tempfile
from zipfile import BadZipFile, ZipFile


BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
IMAGE_ASSETS = {
    "lowres_depth": ".png",
    "lowres_wide": ".png",
    "lowres_wide_intrinsics": ".pincam",
}


def fetch(url, destination):
    if destination.is_file() and destination.stat().st_size:
        return
    partial = destination.with_name(destination.name + ".part")
    subprocess.run(
        ["curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "15",
         "--output", str(partial), url],
        check=True,
    )
    partial.replace(destination)


def extract_asset(scene, asset, suffix):
    target = scene / asset
    archive = scene / f"{asset}.zip"
    with tempfile.TemporaryDirectory(prefix=f".{asset}-", dir=scene) as temporary:
        staging = Path(temporary)
        with ZipFile(archive) as source:
            files = [
                member for member in source.infolist()
                if not member.is_dir() and Path(member.filename).suffix == suffix
                and Path(member.filename).parent == Path(asset)
            ]
            if not files:
                raise ValueError(f"{archive} ne contient aucun fichier {asset}/*{suffix}")
            source.extractall(staging, files)
        (staging / asset).replace(target)
    archive.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="47333462", help="ID numérique du scan")
    parser.add_argument("--split", choices=("Training", "Validation"), default="Training")
    parser.add_argument("--output", type=Path, default=Path("data/arkitscenes"))
    args = parser.parse_args()
    if not args.video_id.isascii() or not args.video_id.isdecimal():
        parser.error("--video-id doit contenir uniquement des chiffres ASCII")

    scene = args.output / "raw" / args.split / args.video_id
    url = f"{BASE_URL}/{args.split}/{args.video_id}"
    try:
        scene.mkdir(parents=True, exist_ok=True)
        for name in (f"{args.video_id}_3dod_annotation.json", "lowres_wide.traj"):
            fetch(f"{url}/{name}", scene / name)
        for asset, suffix in IMAGE_ASSETS.items():
            if not any((scene / asset).glob(f"*{suffix}")):
                fetch(f"{url}/{asset}.zip", scene / f"{asset}.zip")
                extract_asset(scene, asset, suffix)
    except (OSError, ValueError, BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Échec du téléchargement du scan {args.video_id} : {error}\n")

    print(f"Scan prêt : {scene.resolve()}")
    for asset, suffix in IMAGE_ASSETS.items():
        print(f"  {asset} : {sum(1 for _ in (scene / asset).glob(f'*{suffix}'))} fichiers")


if __name__ == "__main__":
    main()
