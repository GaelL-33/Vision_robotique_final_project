"""Lire une scène RGB-D et déprojeter ses pixels valides."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Frame:
    id: str
    rgb_path: Path
    depth_path: Path
    K: np.ndarray
    T_world_cam: np.ndarray

    def read(self):
        with Image.open(self.rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(self.depth_path) as image:
            if image.format != "PNG" or image.mode not in {"I", "I;16", "I;16L", "I;16B"}:
                raise ValueError(f"{self.id}: la profondeur doit être un PNG 16 bits")
            depth = np.asarray(image)
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"{self.id}: le RGB et la profondeur doivent être alignés")
        return rgb, depth


@dataclass(frozen=True)
class Scene:
    id: str
    depth_scale_to_m: float
    frames: tuple[Frame, ...]


def _matrix(values, shape, name):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name}: matrice finie {shape} attendue")
    return matrix


def load_scene(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest["schema_version"] != 1:
        raise ValueError("Version de manifeste attendue : 1")
    scale = float(manifest["depth_scale_to_m"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale_to_m doit être fini et positif")

    frames = []
    seen_ids = set()
    for row in manifest["frames"]:
        frame_id = row["id"]
        if not isinstance(frame_id, str) or not frame_id or frame_id in seen_ids:
            raise ValueError(f"Identifiant de trame invalide ou répété : {frame_id!r}")
        seen_ids.add(frame_id)
        K = _matrix(row["K"], (3, 3), f"{frame_id}/K")
        if (K[0, 0] <= 0 or K[1, 1] <= 0 or K[1, 0] != 0
                or not np.allclose(K[2], [0, 0, 1], atol=1e-8, rtol=0)):
            raise ValueError(f"{frame_id}/K: intrinsèques pinhole attendus")

        pose = _matrix(row["T_world_cam"], (4, 4), f"{frame_id}/T_world_cam")
        rotation = pose[:3, :3]
        if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)):
            raise ValueError(f"{frame_id}/T_world_cam: pose rigide attendue")
        frames.append(Frame(frame_id, path.parent / row["rgb"], path.parent / row["depth"], K, pose))

    if not frames:
        raise ValueError("Le manifeste doit contenir au moins une trame")
    return Scene(manifest["scene_id"], scale, tuple(frames))


def project_depth(depth_m, K, T_world_cam, *, pixel_stride=1):
    """Retourner les coordonnées monde XYZ et les indices image (u, v) associés."""
    if pixel_stride < 1:
        raise ValueError("pixel_stride doit être positif")

    sampled_depth = depth_m[::pixel_stride, ::pixel_stride]
    v, u = np.nonzero(np.isfinite(sampled_depth) & (sampled_depth > 0))
    depths = sampled_depth[v, u]
    pixels = np.column_stack((u, v)) * pixel_stride
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    camera_points = np.linalg.solve(K, homogeneous.T).T * depths[:, None]
    world_points = camera_points @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    return world_points, pixels


def voxel_indices(points, size_m):
    """Indexer la grille au sol avec la même origine pour toutes les vues."""
    if not np.isfinite(size_m) or size_m <= 0:
        raise ValueError("La taille de voxel doit être finie et positive")
    return np.floor(points / size_m).astype(np.int64)

