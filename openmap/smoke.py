"""Fabriquer et vérifier une petite scène à correspondances multi-vues connues."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .rgbd import load_scene, project_depth, voxel_indices


def make_fixture(output):
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "scene_id": "synthetic_two_views",
        "depth_scale_to_m": 0.001,
        "frames": [],
    }
    for index, shift_px in enumerate((0, 3)):
        depth = np.zeros((48, 64), dtype=np.uint16)
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        for x, color in ((10, (240, 70, 35)), (38, (40, 110, 240))):
            region = np.s_[14:34, x - shift_px:x - shift_px + 14]
            depth[region] = 2000
            rgb[region] = color

        frame_id = f"{index:03d}"
        rgb_name = f"rgb_{frame_id}.png"
        depth_name = f"depth_{frame_id}.png"
        Image.fromarray(rgb).save(output / rgb_name)
        Image.fromarray(depth).save(output / depth_name)

        pose = np.eye(4)
        pose[:3, 3] = [0.013 + shift_px * 2.0 / 60.0, 0.013, 0.0]
        config["frames"].append({
            "id": frame_id,
            "rgb": rgb_name,
            "depth": depth_name,
            "K": [[60, 0, 31.5], [0, 60, 23.5], [0, 0, 1]],
            "T_world_cam": pose.tolist(),
        })

    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_smoke(output: Path):
    manifest = make_fixture(output)
    scene = load_scene(manifest)
    point_sets = []
    pixel_sets = []
    color_sets = []
    for frame in scene.frames:
        rgb, depth = frame.read()
        points, pixels = project_depth(depth * scene.depth_scale_to_m, frame.K, frame.T_world_cam)
        if len(points) != 560:
            raise ValueError(f"{frame.id}: 560 points valides attendus, obtenu {len(points)}")
        np.testing.assert_allclose(points[:, 2], 2.0, atol=1e-12, rtol=0)
        point_sets.append(points)
        pixel_sets.append(pixels)
        color_sets.append(rgb[pixels[:, 1], pixels[:, 0]])

    np.testing.assert_array_equal(pixel_sets[0] - pixel_sets[1], np.tile([3, 0], (560, 1)))
    np.testing.assert_array_equal(color_sets[0], color_sets[1])
    np.testing.assert_allclose(point_sets[0], point_sets[1], atol=1e-12, rtol=0)
    voxels = [np.unique(voxel_indices(points, 0.05), axis=0) for points in point_sets]
    np.testing.assert_array_equal(voxels[0], voxels[1])

    return {
        "scene": scene.id,
        "manifest": str(manifest),
        "frames": len(scene.frames),
        "points_per_frame": [len(points) for points in point_sets],
        "overlap_voxels_5_cm": len(voxels[0]),
        "max_world_point_error_m": float(np.linalg.norm(point_sets[0] - point_sets[1], axis=1).max()),
    }

