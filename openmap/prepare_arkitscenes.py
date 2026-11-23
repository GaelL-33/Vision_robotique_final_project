"""Préparer un manifeste RGB-D depuis un scan brut ARKitScenes."""

import argparse
from bisect import bisect_left
import json
import math
import os
from pathlib import Path

import numpy as np

from .rgbd import Frame


def timestamped_files(folder, suffix):
    files = {}
    for path in sorted(folder.glob(f"*{suffix}")):
        timestamp = float(path.stem.rsplit("_", 1)[-1])
        if not math.isfinite(timestamp) or timestamp in files:
            raise ValueError(f"{path} : timestamp invalide ou répété")
        files[timestamp] = path
    if not files:
        raise ValueError(f"Aucun fichier {suffix} dans {folder}")
    return files


def read_trajectory(path):
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] != 7 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError(f"{path} : sept valeurs finies par pose attendues")
    poses = {row[0]: row[1:] for row in rows}
    if len(poses) != len(rows):
        raise ValueError(f"{path} : timestamps de pose répétés")
    return poses


def camera_to_world(pose):
    # Le fichier Apple encode la transformation monde vers caméra.
    x, y, z = pose[:3]
    angle = np.linalg.norm(pose[:3])
    rotation = np.eye(3)
    if angle:
        axis_cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) / angle
        rotation += np.sin(angle) * axis_cross
        rotation += (1 - np.cos(angle)) * (axis_cross @ axis_cross)
    transform = np.eye(4)
    transform[:3, :3] = rotation.T
    transform[:3, 3] = -rotation.T @ pose[3:]
    return transform


def nearest(times, timestamp):
    index = bisect_left(times, timestamp)
    candidates = times[max(0, index - 1):index + 1]
    return min(candidates, key=lambda candidate: abs(candidate - timestamp))


def make_manifest(scan, output, frames, max_pose_gap_ms):
    depths = timestamped_files(scan / "lowres_depth", ".png")
    colors = timestamped_files(scan / "lowres_wide", ".png")
    intrinsics = timestamped_files(scan / "lowres_wide_intrinsics", ".pincam")
    poses = read_trajectory(scan / "lowres_wide.traj")
    pose_times = sorted(poses)

    # Le lecteur Apple associe la couleur et la profondeur par nom de trame.
    # Conserver les paires synchrones évite d'associer des pixels pris en mouvement.
    candidates = []
    for timestamp in sorted(depths.keys() & colors.keys() & intrinsics.keys()):
        pose_time = nearest(pose_times, timestamp)
        gap_ms = abs(pose_time - timestamp) * 1000
        if gap_ms <= max_pose_gap_ms:
            candidates.append((timestamp, pose_time, gap_ms))

    if len(candidates) < frames:
        raise ValueError(f"{len(candidates)} trames synchronisées disponibles pour {frames} demandées")

    entries = []
    gaps = []
    for index in np.linspace(0, len(candidates) - 1, frames, dtype=int):
        timestamp, pose_time, gap_ms = candidates[index]
        rgb, depth = colors[timestamp], depths[timestamp]
        width, height, fx, fy, cx, cy = np.loadtxt(intrinsics[timestamp])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        if not np.isfinite(K).all() or fx <= 0 or fy <= 0:
            raise ValueError(f"{intrinsics[timestamp]} : calibration invalide")
        transform = camera_to_world(poses[pose_time])
        frame = Frame(depth.stem, rgb, depth, K, transform)
        _, depth_image = frame.read()
        if depth_image.shape != (height, width):
            raise ValueError(f"{depth} : dimensions incompatibles avec la calibration")
        entries.append({
            "id": frame.id,
            "rgb": Path(os.path.relpath(rgb, output.parent)).as_posix(),
            "depth": Path(os.path.relpath(depth, output.parent)).as_posix(),
            "K": K.tolist(),
            "T_world_cam": transform.tolist(),
        })
        gaps.append(float(gap_ms))

    manifest = {
        "schema_version": 1,
        "scene_id": scan.name,
        "depth_scale_to_m": 0.001,
        "frames": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest": str(output),
        "scan": str(scan),
        "depth_frames": len(depths),
        "eligible_frames": len(candidates),
        "frame_ids": [entry["id"] for entry in entries],
        "max_pose_gap_ms": max_pose_gap_ms,
        "observed_max_pose_gap_ms": max(gaps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path, help="Dossier du scan brut ARKitScenes")
    parser.add_argument("--output", type=Path, help="Chemin du manifeste à écrire")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--max-pose-gap-ms", type=float, default=2.0)
    args = parser.parse_args()
    if args.frames < 1 or not math.isfinite(args.max_pose_gap_ms) or args.max_pose_gap_ms < 0:
        parser.error("Indiquer un nombre de trames positif et un écart de pose fini et non négatif.")
    output = args.output or Path("runs") / args.scan.name / "manifest.json"
    try:
        result = make_manifest(args.scan, output, args.frames, args.max_pose_gap_ms)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
