"""Produire les aperçus de contrôle RGB-D, multi-vues et annotations ARKitScenes."""

import argparse
from itertools import combinations, product
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .rgbd import load_scene, project_depth


def depth_overlay(rgb, depth, threshold):
    edge = np.zeros(depth.shape, dtype=bool)
    valid = np.isfinite(depth) & (depth > 0)
    edge[:, 1:] = (np.abs(np.diff(depth, axis=1)) > threshold) & valid[:, 1:] & valid[:, :-1]
    edge[1:, :] |= (np.abs(np.diff(depth, axis=0)) > threshold) & valid[1:, :] & valid[:-1, :]
    overlay = rgb.copy()
    overlay[edge] = (255, 36, 160)
    return overlay


def setup_projection(axis, pair):
    x, y = pair
    axis.set(xlabel=f"{'XYZ'[x]} (m)", ylabel=f"{'XYZ'[y]} (m)", aspect="equal")
    axis.grid(alpha=0.2)


def inspect_scene(scene, annotations, selected, output, stride, threshold):
    output.mkdir(parents=True, exist_ok=True)
    rgbd_fig, rgbd_axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    views_fig, views_axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    projections = ((0, 1), (0, 2), (1, 2))
    for axis, pair in zip(views_axes, projections):
        setup_projection(axis, pair)

    clouds, colors = [], []
    for number, frame in enumerate(scene.frames, 1):
        rgb, raw_depth = frame.read()
        depth = raw_depth * scene.depth_scale_to_m
        points, pixels = project_depth(depth, frame.K, frame.T_world_cam, pixel_stride=stride)
        clouds.append(points)
        colors.append(rgb[pixels[:, 1], pixels[:, 0]] / 255)
        if number not in selected:
            continue
        row = selected.index(number)
        images = (rgb, depth, depth_overlay(rgb, depth, threshold))
        titles = ("RGB", "Profondeur (m)", f"Sauts de profondeur > {threshold:g} m")
        for axis, image, title in zip(rgbd_axes[row], images, titles):
            axis.imshow(image, cmap="viridis", interpolation="nearest")
            axis.set_title(f"Trame {number} · {title}")
            axis.axis("off")
        for axis, (x, y) in zip(views_axes, projections):
            axis.scatter(points[:, x], points[:, y], s=2, alpha=0.45,
                         color=("#0072B2", "#D55E00")[row], label=f"Trame {number}")

    views_axes[0].legend(markerscale=4)
    views_fig.suptitle(f"{scene.id} · Superposition des trames")
    rgbd_fig.savefig(output / "controle-rgbd.png", dpi=150)
    views_fig.savefig(output / "controle-multivue.png", dpi=150)
    plt.close(rgbd_fig)
    plt.close(views_fig)

    points, rgb = np.concatenate(clouds), np.concatenate(colors)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for axis, (x, y), title in zip(axes, projections, ("Vue de dessus", "Vue de côté")):
        setup_projection(axis, (x, y))
        axis.set_title(title)
        axis.scatter(points[:, x], points[:, y], c=rgb, s=2, alpha=0.5)

    signs = np.array(list(product((-1, 1), repeat=3)))
    edges = [(a, b) for a, b in combinations(range(8), 2)
             if np.count_nonzero(signs[a] != signs[b]) == 1]
    coverage = []
    for number, annotation in enumerate(annotations, 1):
        box = annotation["segments"]["obbAligned"]
        center = np.asarray(box["centroid"])
        half = np.asarray(box["axesLengths"]) / 2
        rotation = np.asarray(box["normalizedAxes"]).reshape(3, 3)
        corners = (signs * half) @ rotation + center
        inside = np.all(np.abs((points - center) @ rotation.T) <= half, axis=1)
        coverage.append({"box": number, "uid": annotation["uid"],
                         "label": annotation["label"], "points": int(inside.sum())})
        for axis, (x, y) in zip(axes, projections):
            for a, b in edges:
                axis.plot(corners[[a, b], x], corners[[a, b], y], color="#a00065")
            axis.annotate(str(number), (center[x], center[y]), color="#7d004f", weight="bold",
                          bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1})
    fig.suptitle(f"{scene.id} · Boîtes annotées dans le repère du nuage")
    fig.savefig(output / "controle-annotations.png", dpi=150)
    plt.close(fig)
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("annotations", type=Path, help="Fichier Apple *_3dod_annotation.json")
    parser.add_argument("--frames", nargs=2, type=int, default=[1, 2],
                        help="Deux numéros de trames du manifeste, à partir de 1")
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--depth-edge-m", type=float, default=0.15)
    parser.add_argument("--output", type=Path, help="Dossier des aperçus")
    args = parser.parse_args()
    if args.pixel_stride < 1 or not math.isfinite(args.depth_edge_m) or args.depth_edge_m <= 0:
        parser.error("Le pas spatial et le seuil de profondeur doivent être positifs.")
    try:
        scene = load_scene(args.manifest)
        if len(set(args.frames)) != 2 or any(n < 1 or n > len(scene.frames) for n in args.frames):
            raise ValueError("Choisir deux numéros de trames distincts présents dans le manifeste.")
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["data"]
        output = args.output or args.manifest.parent
        coverage = inspect_scene(scene, annotations, args.frames, output,
                                 args.pixel_stride, args.depth_edge_m)
        report = {
            "manifest": str(args.manifest),
            "annotations": str(args.annotations),
            "frame_ids": [scene.frames[n - 1].id for n in args.frames],
            "pixel_stride": args.pixel_stride,
            "depth_edge_m": args.depth_edge_m,
            "boxes": coverage,
        }
        (output / "controle.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Aperçus et comptage par boîte dans {output}")


if __name__ == "__main__":
    main()
