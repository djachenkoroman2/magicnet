"""Dataset adapter for K3DXYZRGB semantic segmentation scenes.

Expected raw file format:
    x y z r g b label

The loader supports mixed `.txt` / `.asc` raw files and can cache them as
`.npy` to avoid repeated text parsing on every epoch.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..build import DATASETS
from ..data_util import crop_pc
from ..k3d_xyz.k3d_xyz import (
    DEFAULT_CMAP,
    read_k3d_xyz_split,
    remap_k3d_xyz_labels,
    resolve_k3d_xyz_path,
)


def read_k3d_xyzrgb_split(split_root, split):
    """Read `train.txt` / `val.txt` / `test.txt` and return scene names."""
    return read_k3d_xyz_split(split_root, split)


def resolve_k3d_xyzrgb_path(raw_root, entry):
    """Resolve a split entry to a real file in `raw/`."""
    return resolve_k3d_xyz_path(raw_root, entry)


def remap_k3d_xyzrgb_labels(labels, label_values):
    """Map raw labels to contiguous indices `0..num_classes-1`."""
    return remap_k3d_xyz_labels(labels, label_values)


def load_k3d_xyzrgb_array(data_path):
    """Load one scene and keep only the first seven columns: xyz + rgb + label."""
    data_path = Path(data_path)
    if data_path.suffix.lower() == '.npy':
        data = np.load(data_path)
    else:
        data = np.loadtxt(data_path, dtype=np.float32)

    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 7:
        raise ValueError(
            f'K3DXYZRGB scene {data_path} must contain at least 7 columns '
            f'`x y z r g b label`, but got shape {data.shape}.'
        )
    return np.asarray(data[:, :7], dtype=np.float32)


@DATASETS.register_module()
class K3DXYZRGB(Dataset):
    """Semantic-segmentation dataset for point clouds with xyz + rgb + class label."""

    gravity_dim = 2

    def __init__(
        self,
        data_root='data/k3d_xyzrgb',
        split='train',
        voxel_size=0.2,
        voxel_max=None,
        transform=None,
        loop=1,
        presample=False,
        variable=False,
        shuffle=True,
        class_names=None,
        label_values=None,
        cache_raw=True,
        **kwargs,
    ):
        super().__init__()

        self.data_root = Path(data_root).expanduser()
        if not self.data_root.is_absolute():
            self.data_root = (Path.cwd() / self.data_root).resolve()
        self.raw_root = self.data_root / 'raw'
        self.split_root = self.data_root / 'splits'
        self.processed_root = self.data_root / 'processed'
        self.cache_root = self.processed_root / 'raw_npy'

        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.loop = loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle
        self.cache_raw = cache_raw

        if not self.raw_root.is_dir():
            raise FileNotFoundError(
                f'K3DXYZRGB raw directory {self.raw_root} does not exist. '
                'Expected raw scenes under `data/k3d_xyzrgb/raw/`.'
            )

        self.data_list = read_k3d_xyzrgb_split(self.split_root, split)
        self.raw_paths = [resolve_k3d_xyzrgb_path(self.raw_root, entry) for entry in self.data_list]

        self.label_values = self._init_label_values(label_values)
        self.class_names = self._init_class_names(class_names)
        self.classes = self.class_names
        self.num_classes = len(self.label_values)
        self.label_to_index = {raw_label: idx for idx, raw_label in enumerate(self.label_values)}
        self.cmap = self._build_cmap(self.num_classes)
        self.num_per_class = self._count_labels_for_split(self.raw_paths)

        if self.presample:
            self.data = self._load_or_build_presampled_cache()
        else:
            self.data = None

        logging.info(
            'Loaded K3DXYZRGB split=%s with %d scenes, num_classes=%d, voxel_size=%s, voxel_max=%s',
            self.split,
            len(self.raw_paths),
            self.num_classes,
            self.voxel_size,
            self.voxel_max,
        )

    def _init_label_values(self, label_values):
        """Resolve the raw label vocabulary."""
        if label_values is not None:
            return [int(value) for value in label_values]

        label_values = set()
        all_entries = set()
        for split_name in ('train', 'val', 'test'):
            split_file = self.split_root / f'{split_name}.txt'
            if not split_file.is_file():
                continue
            all_entries.update(read_k3d_xyzrgb_split(self.split_root, split_name))

        for entry in sorted(all_entries):
            scene = self._load_scene(resolve_k3d_xyzrgb_path(self.raw_root, entry))
            label_values.update(np.unique(scene[:, 6].astype(np.int64)).tolist())

        if not label_values:
            raise ValueError('K3DXYZRGB label_values could not be inferred because no labels were found.')
        return sorted(int(value) for value in label_values)

    def _init_class_names(self, class_names):
        """Return human-readable class names aligned with `label_values`."""
        if class_names is not None:
            class_names = [str(name) for name in class_names]
            if len(class_names) != len(self.label_values):
                raise ValueError(
                    f'K3DXYZRGB class_names length {len(class_names)} does not match '
                    f'label_values length {len(self.label_values)}.'
                )
            return class_names
        return [f'class_{label}' for label in self.label_values]

    def _build_cmap(self, num_classes):
        """Build a color map for logging / visualization outputs."""
        cmap = []
        for idx in range(num_classes):
            cmap.append(DEFAULT_CMAP[idx % len(DEFAULT_CMAP)])
        return cmap

    def _scene_cache_path(self, raw_path):
        """Location of the `.npy` cache for one raw scene."""
        return self.cache_root / f'{raw_path.name}.npy'

    def _load_scene(self, raw_path):
        """Load one raw scene, optionally through a `.npy` cache."""
        raw_path = Path(raw_path)
        if not self.cache_raw:
            return load_k3d_xyzrgb_array(raw_path)

        cache_path = self._scene_cache_path(raw_path)
        if not cache_path.is_file():
            self.cache_root.mkdir(parents=True, exist_ok=True)
            scene = load_k3d_xyzrgb_array(raw_path)
            np.save(cache_path, scene)
            return scene
        return np.load(cache_path)

    def _count_labels_for_split(self, raw_paths):
        """Count points per class for the current split."""
        counts = np.zeros(len(self.label_values), dtype=np.int64)
        for raw_path in raw_paths:
            labels = self._remap_labels(self._load_scene(raw_path)[:, 6])
            uniq, cnt = np.unique(labels, return_counts=True)
            counts[uniq] += cnt
        return counts

    def _presample_cache_path(self):
        """Cache file that stores preprocessed scenes for a fixed setup."""
        voxel_tag = 'none' if self.voxel_size is None else f'{self.voxel_size:.3f}'
        voxel_max_tag = 'none' if self.voxel_max is None else str(self.voxel_max)
        return self.processed_root / f'k3d_xyzrgb_{self.split}_{voxel_tag}_{voxel_max_tag}.pkl'

    def _load_or_build_presampled_cache(self):
        """Build or load static validation/test data."""
        import pickle

        cache_path = self._presample_cache_path()
        if cache_path.is_file():
            with open(cache_path, 'rb') as handle:
                data = pickle.load(handle)
            logging.info('Loaded K3DXYZRGB presampled cache from %s', cache_path)
            return data

        self.processed_root.mkdir(parents=True, exist_ok=True)
        data = []
        for raw_path in self.raw_paths:
            scene = self._load_scene(raw_path)
            coord = scene[:, :3].astype(np.float32)
            coord -= coord.min(0)
            feat = scene[:, 3:6].astype(np.float32)
            label = self._remap_labels(scene[:, 6]).reshape(-1, 1)
            coord, feat, label = crop_pc(
                coord,
                feat,
                label,
                self.split,
                self.voxel_size,
                self.voxel_max,
                downsample=True,
                variable=self.variable,
                shuffle=self.shuffle,
            )
            data.append(np.hstack((coord, feat, label.astype(np.float32))))

        with open(cache_path, 'wb') as handle:
            pickle.dump(data, handle)
        logging.info('Saved K3DXYZRGB presampled cache to %s', cache_path)
        return data

    def _remap_labels(self, raw_labels):
        """Convenience wrapper around the shared label remapper."""
        return remap_k3d_xyzrgb_labels(raw_labels, self.label_values)

    def __getitem__(self, idx):
        """Return one training sample as `{'pos', 'x', 'y', 'heights'}`."""
        data_idx = idx % len(self.raw_paths)
        if self.presample:
            scene = self.data[data_idx]
            coord, feat, label = np.split(scene, [3, 6], axis=1)
        else:
            scene = self._load_scene(self.raw_paths[data_idx])
            coord = scene[:, :3].astype(np.float32)
            coord -= coord.min(0)
            feat = scene[:, 3:6].astype(np.float32)
            label = self._remap_labels(scene[:, 6]).reshape(-1, 1)
            coord, feat, label = crop_pc(
                coord,
                feat,
                label,
                self.split,
                self.voxel_size,
                self.voxel_max,
                downsample=True,
                variable=self.variable,
                shuffle=self.shuffle,
            )

        data = {
            'pos': coord.astype(np.float32),
            'x': feat.astype(np.float32),
            'y': label.astype(np.int64),
        }
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' not in data:
            height = coord[:, self.gravity_dim:self.gravity_dim + 1].astype(np.float32)
            data['heights'] = torch.from_numpy(height)
        return data

    def __len__(self):
        """Dataset length with optional loop-based repetition."""
        return len(self.raw_paths) * self.loop
