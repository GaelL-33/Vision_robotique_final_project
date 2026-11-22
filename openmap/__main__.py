"""Commandes de diagnostic du jalon RGB-D."""

import argparse
import json
from pathlib import Path

import numpy as np

from .rgbd import load_scene, project_depth, voxel_indices
from .smoke import run_smoke


def inspect_scene(manifest, frames, pixel_stride, voxel_size, ply):
    scene = load_scene(manifest)
    summaries = []
    point_sets = []
    color_sets = []
    for frame in scene.frames[:frames]:
        rgb, depth = frame.read()
        xyz, uv = project_depth(depth * scene.depth_scale_to_m, frame.K, frame.T_world_cam,
                                pixel_stride=pixel_stride)
        summary = {
            "id": frame.id,
            "size_px": [rgb.shape[1], rgb.shape[0]],
            "valid_points": len(xyz),
            "camera_center_m": frame.T_world_cam[:3, 3].tolist(),
        }
        if len(xyz):
            summary["bounds_m"] = [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()]
        summaries.append(summary)
        point_sets.append(xyz)
        if ply is not None:
            color_sets.append(rgb[uv[:, 1], uv[:, 0]])

    points = np.concatenate(point_sets)
    occupied = np.unique(voxel_indices(points, voxel_size), axis=0)
    if ply is not None:
        colors = np.concatenate(color_sets)
        vertices = np.empty(len(points), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        vertices["xyz"] = points
        vertices["rgb"] = colors
        header = (
            f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        ply.parent.mkdir(parents=True, exist_ok=True)
        with ply.open("wb") as stream:
            stream.write(header.encode("ascii"))
            stream.write(vertices.tobytes())

    return {
        "scene": scene.id,
        "voxel_size_m": voxel_size,
        "pixel_stride": pixel_stride,
        "occupied_voxels": len(occupied),
        "frames": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Vérifier deux vues RGB-D synthétiques")
    smoke.add_argument("--output", type=Path, default=Path("runs/smoke"))
    inspect = commands.add_parser("inspect", help="Inspecter et exporter les trames d'une scène")
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument("--frames", type=int, default=2)
    inspect.add_argument("--pixel-stride", type=int, default=1)
    inspect.add_argument("--voxel-size", type=float, default=0.05)
    inspect.add_argument("--ply", type=Path)
    args = parser.parse_args()

    try:
        if args.command == "smoke":
            result = run_smoke(args.output)
        else:
            if args.frames < 1:
                raise ValueError("--frames doit être positif")
            result = inspect_scene(args.manifest, args.frames, args.pixel_stride,
                                   args.voxel_size, args.ply)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

