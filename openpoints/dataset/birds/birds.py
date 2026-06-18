import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..build import DATASETS


def load_birds_xyz_array(file_path):
    """Load one point cloud from a whitespace-delimited `x y z` text file."""
    data = np.loadtxt(file_path, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError(
            f'Birds point cloud {file_path} must contain at least 3 columns `x y z`, '
            f'but got shape {data.shape}.'
        )
    return np.asarray(data[:, :3], dtype=np.float32)


@DATASETS.register_module()
class Birds(Dataset):
    """Point-cloud classification dataset backed by `data/birds/<class>/*.txt`."""

    gravity_dim = 2
    split_names = ('train', 'val', 'test')

    def __init__(
        self,
        data_root='data/birds',
        split='train',
        num_points=1024,
        transform=None,
        split_ratios=(0.8, 0.1, 0.1),
        seed=0,
        normalize=False,
        cache_data=False,
        **kwargs,
    ):
        super().__init__()

        self.data_root = Path(data_root).expanduser()
        if not self.data_root.is_absolute():
            self.data_root = (Path.cwd() / self.data_root).resolve()
        self.split_root = self.data_root / 'splits'
        self.split = split.lower()
        if self.split not in (*self.split_names, 'all'):
            raise ValueError(
                f'Unsupported Birds split `{split}`. Expected one of '
                f'{self.split_names + ("all",)}.'
            )

        self.num_points = None if num_points is None else int(num_points)
        if self.num_points is not None and self.num_points <= 0:
            raise ValueError(f'`num_points` must be positive or None, got {num_points}.')

        self.transform = transform
        self.seed = int(seed)
        self.normalize = bool(normalize)
        self.cache_data = bool(cache_data)
        self._cache = {} if self.cache_data else None

        self.classes = self._discover_classes()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = self._build_samples(split_ratios)

        if not self.samples:
            raise ValueError(
                f'Birds split `{self.split}` is empty under {self.data_root}. '
                'Check `split_ratios` or provide explicit split files.'
            )

        logging.info(
            'Loaded Birds split=%s with %d samples, num_classes=%d, num_points=%s',
            self.split,
            len(self.samples),
            self.num_classes,
            self.num_points,
        )

    @property
    def num_classes(self):
        return len(self.classes)

    def _discover_classes(self):
        class_dirs = sorted(
            path for path in self.data_root.iterdir()
            if path.is_dir() and path.name != 'splits'
        )
        classes = [path.name for path in class_dirs]
        if not classes:
            raise FileNotFoundError(
                f'Birds dataset root {self.data_root} does not contain any class folders.'
            )
        return classes

    def _build_samples(self, split_ratios):
        explicit_split_files = {
            split_name: self.split_root / f'{split_name}.txt'
            for split_name in self.split_names
        }
        has_explicit_splits = any(path.is_file() for path in explicit_split_files.values())

        if has_explicit_splits:
            return self._build_samples_from_split_files(explicit_split_files)
        return self._build_samples_from_ratios(split_ratios)

    def _build_samples_from_split_files(self, split_files):
        if self.split == 'all':
            requested_paths = [path for path in split_files.values() if path.is_file()]
        else:
            split_file = split_files[self.split]
            if not split_file.is_file():
                raise FileNotFoundError(
                    f'Birds split file {split_file} does not exist. '
                    'Expected explicit split files under `data_root/splits/`.'
                )
            requested_paths = [split_file]

        seen = set()
        samples = []
        for split_file in requested_paths:
            for line in split_file.read_text().splitlines():
                entry = line.strip()
                if not entry:
                    continue
                file_path = self._resolve_split_entry(entry)
                if file_path in seen:
                    continue
                seen.add(file_path)
                class_name = file_path.parent.name
                if class_name not in self.class_to_idx:
                    raise ValueError(
                        f'Split entry `{entry}` resolved to {file_path}, '
                        f'but `{class_name}` is not a known Birds class.'
                    )
                samples.append((file_path, self.class_to_idx[class_name]))
        return samples

    def _resolve_split_entry(self, entry):
        entry_path = Path(entry).expanduser()
        candidates = []
        if entry_path.is_absolute():
            candidates.append(entry_path)
            if not entry_path.suffix:
                candidates.append(entry_path.with_suffix('.txt'))
        else:
            candidates.append(self.data_root / entry_path)
            if not entry_path.suffix:
                candidates.append((self.data_root / entry_path).with_suffix('.txt'))

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        raise FileNotFoundError(
            f'Birds split entry `{entry}` was not found under {self.data_root}.'
        )

    def _build_samples_from_ratios(self, split_ratios):
        ratios = np.asarray(split_ratios, dtype=np.float64)
        if ratios.shape != (3,):
            raise ValueError(
                f'`split_ratios` must contain exactly 3 values for train/val/test, got {split_ratios}.'
            )
        if np.any(ratios < 0):
            raise ValueError(f'`split_ratios` must be non-negative, got {split_ratios}.')
        ratio_sum = float(ratios.sum())
        if ratio_sum <= 0:
            raise ValueError(f'`split_ratios` must sum to a positive value, got {split_ratios}.')
        ratios = ratios / ratio_sum

        samples = []
        for class_name in self.classes:
            class_dir = self.data_root / class_name
            class_files = sorted(class_dir.glob('*.txt'))
            if not class_files:
                logging.warning('Skipping empty Birds class folder %s', class_dir)
                continue

            rng = np.random.default_rng(self.seed + self.class_to_idx[class_name])
            shuffled_indices = rng.permutation(len(class_files))
            class_files = [class_files[idx] for idx in shuffled_indices]

            counts = self._compute_split_counts(len(class_files), ratios)
            train_end = counts[0]
            val_end = counts[0] + counts[1]
            split_to_files = {
                'train': class_files[:train_end],
                'val': class_files[train_end:val_end],
                'test': class_files[val_end:],
                'all': class_files,
            }

            for file_path in split_to_files[self.split]:
                samples.append((file_path, self.class_to_idx[class_name]))
        return samples

    def _compute_split_counts(self, num_items, ratios):
        raw_counts = ratios * num_items
        counts = np.floor(raw_counts).astype(np.int64)
        remainder = int(num_items - counts.sum())
        if remainder > 0:
            priorities = np.argsort(-(raw_counts - counts))
            for idx in priorities[:remainder]:
                counts[idx] += 1
        return counts

    def _load_points(self, file_path):
        if self.cache_data:
            cached = self._cache.get(file_path)
            if cached is None:
                cached = load_birds_xyz_array(file_path)
                self._cache[file_path] = cached
            return cached.copy()
        return load_birds_xyz_array(file_path)

    def _resample_points(self, points):
        if self.num_points is None:
            sampled = points
        else:
            num_available = points.shape[0]
            if num_available == 0:
                raise ValueError('Birds point cloud cannot be empty.')
            if num_available >= self.num_points:
                if self.split == 'train':
                    choice = np.random.choice(num_available, self.num_points, replace=False)
                else:
                    choice = np.linspace(0, num_available - 1, self.num_points, dtype=np.int64)
                sampled = points[choice]
            else:
                if self.split == 'train':
                    extra = np.random.choice(num_available, self.num_points - num_available, replace=True)
                else:
                    extra = np.arange(self.num_points - num_available, dtype=np.int64) % num_available
                choice = np.concatenate([np.arange(num_available, dtype=np.int64), extra])
                sampled = points[choice]

        if self.split == 'train':
            np.random.shuffle(sampled)
        return np.asarray(sampled, dtype=np.float32)

    def _normalize_points(self, points):
        centroid = points.mean(axis=0, keepdims=True)
        points = points - centroid
        scale = np.linalg.norm(points, axis=1).max()
        if scale > 0:
            points = points / scale
        return points.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]
        points = self._load_points(file_path)
        points = self._resample_points(points)
        if self.normalize:
            points = self._normalize_points(points)

        data = {
            'pos': points.astype(np.float32),
            'y': np.int64(label),
        }
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' in data:
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']
        return data

