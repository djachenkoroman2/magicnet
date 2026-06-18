"""Dataset adapter for K3DXYZ semantic segmentation scenes.

Expected raw file format:
    x y z label

The loader supports mixed `.txt` / `.asc` raw files and can cache them as `.npy`
to avoid repeated text parsing on every epoch.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..build import DATASETS
from ..data_util import crop_pc

# Raw scene names in split files may refer to any of these formats.
SUPPORTED_SUFFIXES = ('.txt', '.asc', '.npy')

# Fallback colors used by visualization helpers and metric dumps.
DEFAULT_CMAP = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
]


def read_k3d_xyz_split(split_root, split):
    """Read `train.txt` / `val.txt` / `test.txt` and return scene names."""
    split_path = Path(split_root) / f'{split}.txt'
    if not split_path.is_file():
        raise FileNotFoundError(
            f'K3DXYZ split file {split_path} does not exist. '
            'Expected split files under `data/k3d_xyz/splits/`.'
        )
    return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]


def resolve_k3d_xyz_path(raw_root, entry):
    """Resolve a split entry to a real file in `raw/`.

    Split files may contain names with or without an extension, so we try a
    small set of supported suffixes.
    """
    raw_root = Path(raw_root)
    entry = Path(entry)

    candidates = [raw_root / entry]
    if entry.suffix:
        candidates.extend(raw_root / f'{entry.stem}{suffix}' for suffix in SUPPORTED_SUFFIXES)
    else:
        candidates.extend(raw_root / f'{entry.name}{suffix}' for suffix in SUPPORTED_SUFFIXES)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f'K3DXYZ scene `{entry}` was not found under {raw_root}. '
        f'Tried suffixes: {", ".join(SUPPORTED_SUFFIXES)}'
    )


def load_k3d_xyz_array(data_path):
    """Load one scene and keep only the first four columns: xyz + label."""
    data_path = Path(data_path)
    if data_path.suffix.lower() == '.npy':
        data = np.load(data_path)
    else:
        data = np.loadtxt(data_path, dtype=np.float32)

    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 4:
        raise ValueError(
            f'K3DXYZ scene {data_path} must contain at least 4 columns `x y z label`, '
            f'but got shape {data.shape}.'
        )
    return np.asarray(data[:, :4], dtype=np.float32)


def remap_k3d_xyz_labels(labels, label_values):
    """Map raw labels to contiguous indices `0..num_classes-1`.

    This keeps the training pipeline compatible with PyTorch losses and the
    confusion-matrix code, which both assume dense class ids.
    """
    label_values = np.asarray(label_values, dtype=np.int64)
    lookup = {int(raw_label): idx for idx, raw_label in enumerate(label_values.tolist())}
    try:
        return np.vectorize(lookup.__getitem__, otypes=[np.int64])(labels.astype(np.int64))
    except KeyError as exc:
        raise ValueError(
            f'Encountered unexpected raw label {exc.args[0]} in K3DXYZ. '
            f'Expected one of {label_values.tolist()}.'
        ) from exc


@DATASETS.register_module()
class K3DXYZ(Dataset):
    """Semantic-segmentation dataset for point clouds with xyz + class label."""

    # The z axis is treated as "height" when `feature_keys` includes `heights`.
    gravity_dim = 2

    def __init__(
        self,
        data_root='data/k3d_xyz',
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

        # Normalize paths once so the dataset works from any current directory.
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

        # Raw scenes and split files are required for this dataset.
        if not self.raw_root.is_dir():
            raise FileNotFoundError(
                f'K3DXYZ raw directory {self.raw_root} does not exist. '
                'Expected raw scenes under `data/k3d_xyz/raw/`.'
            )

        # Split files decide which scenes belong to train / val / test.
        self.data_list = read_k3d_xyz_split(self.split_root, split)
        self.raw_paths = [resolve_k3d_xyz_path(self.raw_root, entry) for entry in self.data_list]

        # `label_values` stores raw ids from the dataset, while the model sees
        # only compact indices `0..num_classes-1`.
        self.label_values = self._init_label_values(label_values)
        self.class_names = self._init_class_names(class_names)
        self.classes = self.class_names
        self.num_classes = len(self.label_values)
        self.label_to_index = {raw_label: idx for idx, raw_label in enumerate(self.label_values)}
        self.cmap = self._build_cmap(self.num_classes)
        self.num_per_class = self._count_labels_for_split(self.raw_paths)

        # Optional presampling stores already voxelized / cropped scenes in a
        # pickle cache. This speeds up validation for static settings.
        if self.presample:
            self.data = self._load_or_build_presampled_cache()
        else:
            self.data = None

        logging.info(
            'Loaded K3DXYZ split=%s with %d scenes, num_classes=%d, voxel_size=%s, voxel_max=%s',
            self.split,
            len(self.raw_paths),
            self.num_classes,
            self.voxel_size,
            self.voxel_max,
        )

    def _init_label_values(self, label_values):
        """Resolve the raw label vocabulary.

        If the config already provides label ids, we trust it. Otherwise we scan
        all available split files and infer the unique labels from disk.
        """
        if label_values is not None:
            return [int(value) for value in label_values]

        label_values = set()
        all_entries = set()
        for split_name in ('train', 'val', 'test'):
            split_file = self.split_root / f'{split_name}.txt'
            if not split_file.is_file():
                continue
            all_entries.update(read_k3d_xyz_split(self.split_root, split_name))

        for entry in sorted(all_entries):
            scene = self._load_scene(resolve_k3d_xyz_path(self.raw_root, entry))
            label_values.update(np.unique(scene[:, 3].astype(np.int64)).tolist())

        if not label_values:
            raise ValueError('K3DXYZ label_values could not be inferred because no labels were found.')
        return sorted(int(value) for value in label_values)

    def _init_class_names(self, class_names):
        """Return human-readable class names aligned with `label_values`."""
        if class_names is not None:
            class_names = [str(name) for name in class_names]
            if len(class_names) != len(self.label_values):
                raise ValueError(
                    f'K3DXYZ class_names length {len(class_names)} does not match '
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
        """Load one raw scene, optionally through a `.npy` cache.

        Text parsing is expensive, so the first read can materialize a cached
        binary version under `processed/raw_npy/`.
        """
        raw_path = Path(raw_path)
        if not self.cache_raw:
            return load_k3d_xyz_array(raw_path)

        cache_path = self._scene_cache_path(raw_path)
        if not cache_path.is_file():
            self.cache_root.mkdir(parents=True, exist_ok=True)
            scene = load_k3d_xyz_array(raw_path)
            np.save(cache_path, scene)
            return scene
        return np.load(cache_path)

    def _count_labels_for_split(self, raw_paths):
        """Count points per class for the current split.

        These counts are useful for weighted losses and for understanding class
        imbalance in the dataset.
        """
        counts = np.zeros(len(self.label_values), dtype=np.int64)
        for raw_path in raw_paths:
            labels = self._remap_labels(self._load_scene(raw_path)[:, 3])
            uniq, cnt = np.unique(labels, return_counts=True)
            counts[uniq] += cnt
        return counts

    def _presample_cache_path(self):
        """Cache file that stores preprocessed scenes for a fixed setup."""
        voxel_tag = 'none' if self.voxel_size is None else f'{self.voxel_size:.3f}'
        voxel_max_tag = 'none' if self.voxel_max is None else str(self.voxel_max)
        return self.processed_root / f'k3d_xyz_{self.split}_{voxel_tag}_{voxel_max_tag}.pkl'

    def _load_or_build_presampled_cache(self):
        """Build or load static validation/test data.

        Each cached scene already contains cropped coordinates and remapped
        labels, so later dataset reads avoid repeating the same preprocessing.
        """
        import pickle

        cache_path = self._presample_cache_path()
        if cache_path.is_file():
            with open(cache_path, 'rb') as handle:
                data = pickle.load(handle)
            logging.info('Loaded K3DXYZ presampled cache from %s', cache_path)
            return data

        self.processed_root.mkdir(parents=True, exist_ok=True)
        data = []
        for raw_path in self.raw_paths:
            scene = self._load_scene(raw_path)
            coord = scene[:, :3].astype(np.float32)
            coord -= coord.min(0)
            label = self._remap_labels(scene[:, 3]).reshape(-1, 1)
            coord, _, label = crop_pc(
                coord,
                None,
                label,
                self.split,
                self.voxel_size,
                self.voxel_max,
                downsample=True,
                variable=self.variable,
                shuffle=self.shuffle,
            )
            data.append(np.hstack((coord, label.astype(np.float32))))

        with open(cache_path, 'wb') as handle:
            pickle.dump(data, handle)
        logging.info('Saved K3DXYZ presampled cache to %s', cache_path)
        return data

    def _remap_labels(self, raw_labels):
        """Convenience wrapper around the shared label remapper."""
        return remap_k3d_xyz_labels(raw_labels, self.label_values)

    def __getitem__(self, idx):
        """Return one training sample as `{'pos', 'y', 'heights'}`.

        `pos`:
            XYZ coordinates after optional voxel downsampling / cropping.
        `y`:
            Per-point class ids in the compact `0..num_classes-1` space.
        `heights`:
            One extra feature channel derived from the z coordinate.
        """
        data_idx = idx % len(self.raw_paths)
        if self.presample:
            # Presampled cache already stores post-crop xyz + label.
            scene = self.data[data_idx]
            coord, label = scene[:, :3], scene[:, 3:4]
        else:
            # Training mode loads the raw scene and crops it on the fly so each
            # epoch can see slightly different point subsets.
            scene = self._load_scene(self.raw_paths[data_idx])
            coord = scene[:, :3].astype(np.float32)
            coord -= coord.min(0)
            label = self._remap_labels(scene[:, 3]).reshape(-1, 1)
            coord, _, label = crop_pc(
                coord,
                None,
                label,
                self.split,
                self.voxel_size,
                self.voxel_max,
                downsample=True,
                variable=self.variable,
                shuffle=self.shuffle,
            )

        # The segmentation pipeline expects a dictionary with point positions and
        # per-point labels. Features are assembled later via `feature_keys`.
        data = {
            'pos': coord.astype(np.float32),
            'y': label.astype(np.int64),
        }
        if self.transform is not None:
            data = self.transform(data)

        # Add a simple geometric feature if the transform stack did not create
        # one. This keeps PointNet++ usable even without RGB/intensity inputs.
        if 'heights' not in data:
            height = coord[:, self.gravity_dim:self.gravity_dim + 1].astype(np.float32)
            data['heights'] = torch.from_numpy(height)
        return data

    def __len__(self):
        """Dataset length with optional loop-based repetition."""
        return len(self.raw_paths) * self.loop
