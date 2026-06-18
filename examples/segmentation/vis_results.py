#!/usr/bin/env python
# coding: utf-8

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDX = 0
DATA_DIR = PROJECT_ROOT / "pretrained" / "pix4point" / "mae-s" / "visualization"
DATASET_NAME = "s3dissphere"
ROOF_HEIGHT = 3
METHOD_NAMES = ("input", "pix4point", "gt")


def load_vis3d_module():
    vis3d_path = PROJECT_ROOT / "openpoints" / "dataset" / "vis3d.py"
    spec = importlib.util.spec_from_file_location("vis3d_standalone", vis3d_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load visualization helpers from {vis3d_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_colors(colors):
    if colors.dtype.kind in {"u", "i"} or np.max(colors) > 1.0:
        return colors.astype(np.float32) / 255.0
    return colors.astype(np.float32)


def main():
    vis3d = load_vis3d_module()
    points_list = []
    colors_list = []
    valid_idx = None

    for method_name in METHOD_NAMES:
        file_path = DATA_DIR / f"{method_name}-{DATASET_NAME}-{IDX}.obj"
        if not file_path.is_file():
            raise FileNotFoundError(f"Visualization file not found: {file_path}")

        points, colors = vis3d.read_obj(file_path)

        if valid_idx is None:
            valid_idx = points[:, 2] < ROOF_HEIGHT
        elif len(points) != len(valid_idx):
            raise ValueError(
                f"Point count mismatch for {file_path}: "
                f"expected {len(valid_idx)}, got {len(points)}"
            )

        points_list.append(points[valid_idx])
        colors_list.append(normalize_colors(colors[valid_idx]))

    vis3d.vis_multi_points(points_list, colors_list)


if __name__ == "__main__":
    main()
