import logging
import os
import os.path as osp
import pickle
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


@DATASETS.register_module()
class ScanObjectNNHardest(Dataset):
    """The hardest variant of ScanObjectNN. 
    The data we use is: `training_objectdataset_augmentedrot_scale75.h5`[1], 
    where there are 2048 points in training and testing.
    The number of training samples is: 11416, and the number of testing samples is 2882. 
    Args:
    """
    classes = [
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "display",
        "door",
        "shelf",
        "table",
        "bed",
        "pillow",
        "sink",
        "sofa",
        "toilet",
    ]
    num_classes = 15
    gravity_dim = 1
    archive_name = 'ScanObjectNN.tar'
    download_url = 'https://drive.google.com/uc?id=1iM3mhMJ_N0x5pytcP831l3ZFwbLmbwzi'
    required_files = (
        'training_objectdataset_augmentedrot_scale75.h5',
        'test_objectdataset_augmentedrot_scale75.h5',
    )

    @classmethod
    def _normalize_data_dir(cls, data_dir):
        data_dir = Path(data_dir).expanduser()
        if not data_dir.is_absolute():
            data_dir = (Path.cwd() / data_dir).resolve()

        candidates = [data_dir]
        if data_dir.name == 'ScanObjectNN':
            candidates.append(data_dir / 'h5_files' / 'main_split')
        elif data_dir.name == 'h5_files':
            candidates.append(data_dir / 'main_split')
        elif data_dir.name != 'main_split':
            candidates.append(data_dir / 'h5_files' / 'main_split')

        for candidate in candidates:
            if any((candidate / filename).is_file() for filename in cls.required_files):
                return candidate

        for candidate in candidates:
            if candidate.name == 'main_split':
                return candidate

        return candidates[-1]

    @classmethod
    def _download_root(cls, data_dir):
        if data_dir.name == 'main_split' and data_dir.parent.name == 'h5_files':
            return data_dir.parent.parent.parent
        if data_dir.name == 'h5_files':
            return data_dir.parent
        if data_dir.name == 'ScanObjectNN':
            return data_dir.parent
        return data_dir.parent

    @classmethod
    def _ensure_dataset_available(cls, data_dir, download):
        missing_files = [filename for filename in cls.required_files if not (data_dir / filename).is_file()]
        if not missing_files and data_dir.is_dir():
            return

        if download:
            download_root = cls._download_root(data_dir)
            archive_path = download_root / cls.archive_name
            download_root.mkdir(parents=True, exist_ok=True)

            try:
                import gdown
            except ImportError as exc:
                raise FileNotFoundError(
                    f'{data_dir} is missing required ScanObjectNN files {missing_files}. '
                    'Install `gdown` or run `bash script/download_scanobjectnn.sh` to prepare the dataset.'
                ) from exc

            if not archive_path.is_file():
                logging.warning(
                    'ScanObjectNN hardest split was not found in %s. Downloading it to %s',
                    data_dir,
                    archive_path,
                )
                gdown.download(cls.download_url, str(archive_path), quiet=False)
            else:
                logging.warning(
                    'ScanObjectNN archive already exists at %s. Reusing it to prepare the dataset.',
                    archive_path,
                )

            with tarfile.open(archive_path) as archive:
                archive.extractall(download_root)

            missing_files = [filename for filename in cls.required_files if not (data_dir / filename).is_file()]

        if missing_files:
            raise FileNotFoundError(
                f'{data_dir} is missing required ScanObjectNN files {missing_files}. '
                'Run `bash script/download_scanobjectnn.sh` or download the hardest split into '
                '`data/ScanObjectNN/h5_files/main_split/`.'
            )

    def __init__(self, data_dir, split,
                 num_points=2048,
                 uniform_sample=True,
                 transform=None,
                 download=True,
                 **kwargs):
        super().__init__()
        self.partition = split
        self.transform = transform
        self.num_points = num_points
        data_dir = self._normalize_data_dir(data_dir)
        self._ensure_dataset_available(data_dir, download)
        split_name = 'training' if split == 'train' else 'test'
        h5_name = data_dir / f'{split_name}_objectdataset_augmentedrot_scale75.h5'

        with h5py.File(h5_name, 'r') as f:
            self.points = np.array(f['data']).astype(np.float32)
            self.labels = np.array(f['label']).astype(int)

        if split_name == 'test' and uniform_sample:
            precomputed_path = data_dir / f'{split_name}_objectdataset_augmentedrot_scale75_1024_fps.pkl'
            if not precomputed_path.exists():
                points = torch.from_numpy(self.points).to(torch.float32).cuda()
                self.points = fps(points, 1024).cpu().numpy()
                with open(precomputed_path, 'wb') as f:
                    pickle.dump(self.points, f)
                    print(f"{precomputed_path} saved successfully")
            else:
                with open(precomputed_path, 'rb') as f:
                    self.points = pickle.load(f)
                    print(f"{precomputed_path} load successfully")
        logging.info(f'Successfully load ScanObjectNN {split} '
                     f'size: {self.points.shape}, num_classes: {self.labels.max()+1}')

    @property
    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, idx):
        current_points = self.points[idx][:self.num_points]
        label = self.labels[idx]
        if self.partition == 'train':
            np.random.shuffle(current_points)
        data = {'pos': current_points,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        # height appending. @KPConv
        # TODO: remove pos here, and only use heights. 
        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = torch.cat((data['pos'],
                                   torch.from_numpy(current_points[:, self.gravity_dim:self.gravity_dim+1] - current_points[:, self.gravity_dim:self.gravity_dim+1].min())), dim=1)
        return data

    def __len__(self):
        return self.points.shape[0]

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """
