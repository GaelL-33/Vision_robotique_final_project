"""Installer et vérifier l'environnement d'inférence du projet."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent
SAM_COMMIT = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
SAM_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
SAM_URL = f"https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/{SAM_COMMIT}/weights/mobile_sam.pt"


def check_checkpoint(path):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SAM_SHA256:
            raise ValueError(f"SHA-256 inattendu pour {path}")


def download_checkpoint(path):
    if path.exists():
        check_checkpoint(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".pt.part")
    with urlopen(SAM_URL, timeout=60) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    check_checkpoint(partial)
    partial.replace(path)


def verify(backend, weights):
    import numpy as np
    import open_clip
    from PIL import Image
    import torch
    import torchvision
    from mobile_sam import SamPredictor, sam_model_registry

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible pour PyTorch. Vérifier le pilote NVIDIA avec nvidia-smi.")
    if not torchvision.extension._has_ops():
        raise RuntimeError("Les opérations natives de torchvision sont indisponibles.")

    config = json.loads((ROOT / "configs/pilot.json").read_text(encoding="utf-8"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.ones(4, device=backend).sum().item()
    print(f"PyTorch {torch.__version__}, torchvision {torchvision.__version__}, backend {backend}")
    if backend == "cuda":
        print(torch.cuda.get_device_name())

    checkpoint = Path(weights)
    check_checkpoint(checkpoint)
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[16:48, 16:48] = [240, 70, 35]
    with torch.inference_mode():
        sam = sam_model_registry[config["mobile_sam"]["model_type"]](checkpoint=str(checkpoint))
        predictor = SamPredictor(sam.to(backend).eval())
        predictor.set_image(pixels)
        masks, scores, _ = predictor.predict(
            box=np.array([16, 16, 47, 47]), multimask_output=False
        )
        if masks.shape != (1, 64, 64) or not np.isfinite(scores).all():
            raise RuntimeError("Sortie MobileSAM invalide.")
        print(f"MobileSAM  masque {masks.shape}")
        del predictor, sam
        if backend == "cuda":
            torch.cuda.empty_cache()

        clip = config["clip"]
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip["model"], pretrained=clip["pretrained"], device=backend
        )
        model.eval()
        crop = Image.fromarray(pixels).crop((16, 16, 48, 48))
        tokens = open_clip.get_tokenizer(clip["model"])(["a photo of a red square"])
        image = model.encode_image(preprocess(crop).unsqueeze(0).to(backend))
        text = model.encode_text(tokens.to(backend))
        if image.shape != (1, 512) or text.shape != image.shape:
            raise RuntimeError("Dimensions OpenCLIP inattendues.")
        if not torch.isfinite(image).all() or not torch.isfinite(text).all():
            raise RuntimeError("Descripteurs OpenCLIP non finis.")
        print(f"OpenCLIP  image {tuple(image.shape)}, texte {tuple(text.shape)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--venv", type=Path, help="Chemin de l'environnement Python")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--hf-home", type=Path, default=os.environ.get("HF_HOME", ROOT / "cache/huggingface"))
    parser.add_argument("--tmp-dir", type=Path, help="Répertoire temporaire de pip")
    parser.add_argument("--check-only", action="store_true", help="Vérifier hors ligne sans installer")
    args = parser.parse_args()

    environment = args.venv or ROOT / (".venv-cuda" if args.backend == "cuda" else ".venv")
    environment = environment.absolute()
    bin_dir, executable = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    python = environment / bin_dir / executable
    checkpoint = args.weights_dir.absolute() / "mobile_sam.pt"
    env = os.environ.copy()
    env["HF_HOME"] = str(args.hf_home.absolute())
    try:
        if args.check_only:
            env["HF_HUB_OFFLINE"] = "1"
            if not python.is_file():
                raise FileNotFoundError(f"Environnement absent : {environment}")
        else:
            if args.tmp_dir:
                temporary = args.tmp_dir.absolute()
                temporary.mkdir(parents=True, exist_ok=True)
                env.update(TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary))
            if not python.is_file():
                subprocess.run([sys.executable, "-m", "venv", str(environment.resolve())],
                               env=env, check=True)

            pip = [str(python), "-m", "pip"]
            torch_packages = ["torch==2.7.1", "torchvision==0.22.1"]
            if sys.platform != "darwin" or args.backend == "cuda":
                variant = "cu128" if args.backend == "cuda" else "cpu"
                torch_packages = [f"{package}+{variant}" for package in torch_packages]
                torch_packages += ["--index-url", f"https://download.pytorch.org/whl/{variant}"]
            subprocess.run([*pip, "install", "--no-cache-dir", *torch_packages], env=env, check=True)
            subprocess.run(
                [*pip, "install", "--no-cache-dir", "-r", "requirements.txt",
                 "-r", "requirements-models.txt", "-r", "requirements-analysis.txt"],
                cwd=ROOT, env=env, check=True,
            )
            download_checkpoint(checkpoint)

        subprocess.run([str(python), "-m", "pip", "check"], env=env, check=True)
        subprocess.run(
            [str(python), "-c", "import sys; from openmap.setup_models import verify; verify(*sys.argv[1:])",
             args.backend, str(checkpoint)],
            cwd=ROOT, env=env, check=True,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Configuration modèles échouée : {error}\n")
    print(f"Configuration vérifiée avec {python}")


if __name__ == "__main__":
    main()
