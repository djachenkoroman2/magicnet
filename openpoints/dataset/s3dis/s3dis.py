import os
import pickle
import logging
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from ..data_util import crop_pc, voxelize
from ..build import DATASETS


@DATASETS.register_module()
class S3DIS(Dataset):
    classes = ['ceiling',
               'floor',
               'wall',
               'beam',
               'column',
               'window',
               'door',
               'chair',
               'table',
               'bookcase',
               'sofa',
               'board',
               'clutter']
    num_classes = 13
    num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                              650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
    class2color = {'ceiling':     [0, 255, 0],
                   'floor':       [0, 0, 255],
                   'wall':        [0, 255, 255],
                   'beam':        [255, 255, 0],
                   'column':      [255, 0, 255],
                   'window':      [100, 100, 255],
                   'door':        [200, 200, 100],
                   'table':       [170, 120, 200],
                   'chair':       [255, 0, 0],
                   'sofa':        [200, 100, 100],
                   'bookcase':    [10, 200, 100],
                   'board':       [200, 200, 200],
                   'clutter':     [50, 50, 50]}
    cmap = [*class2color.values()]
    gravity_dim = 2
    archive_name = 's3disfull.tar'
    download_url = 'https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y'
    """S3DIS dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (794855.5, 1005913.0147058824, 939501.4733064277)
    Args:
        data_root (str, optional): Defaults to 'data/S3DIS/s3disfull'.
        test_area (int, optional): Defaults to 5.
        voxel_size (float, optional): the voxel size for donwampling. Defaults to 0.04.
        voxel_max (_type_, optional): subsample the max number of point per point cloud. Set None to use all points.  Defaults to None.
        split (str, optional): Defaults to 'train'.
        transform (_type_, optional): Defaults to None.
        loop (int, optional): split loops for each epoch. Defaults to 1.
        presample (bool, optional): wheter to downsample each point cloud before training. Set to False to downsample on-the-fly. Defaults to False.
        variable (bool, optional): where to use the original number of points. The number of point per point cloud is variable. Defaults to False.
    """
    def __init__(self,
                 data_root: str = 'data/S3DIS/s3disfull',
                 test_area: int = 5,
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 split: str = 'train',
                 transform=None,
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 shuffle: bool = True,
                 download: bool = True,
                 ):

        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle

        data_root = self._normalize_data_root(data_root)
        self.data_root = str(data_root)
        raw_root = data_root / 'raw'
        self._ensure_dataset_available(data_root, download)
        self.raw_root = str(raw_root)
        data_list = sorted(os.listdir(raw_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [
                item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [
                item for item in data_list if 'Area_{}'.format(test_area) in item]

        processed_root = os.path.join(str(data_root), 'processed')
        filename = os.path.join(
            processed_root, f's3dis_{split}_area{test_area}_{voxel_size:.3f}_{str(voxel_max)}.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading S3DISFull {split} split on Test Area {test_area}'):
                data_path = os.path.join(raw_root, item + '.npy')
                cdata = np.load(data_path).astype(np.float32)
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                if voxel_size:
                    coord, feat, label = cdata[:,0:3], cdata[:, 3:6], cdata[:, 6:7]
                    uniq_idx = voxelize(coord, voxel_size)
                    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
                    cdata = np.hstack((coord, feat, label))
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        if self.presample:
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        else:
            data_path = os.path.join(
                self.raw_root, self.data_list[data_idx] + '.npy')
            cdata = np.load(data_path).astype(np.float32)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)
            # TODO: do we need to -np.min in cropped data?
        label = label.squeeze(-1).astype(np.int64)
        data = {'pos': coord, 'x': feat, 'y': label}
        # pre-process.
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' not in data.keys():
            data['heights'] =  torch.from_numpy(coord[:, self.gravity_dim:self.gravity_dim+1].astype(np.float32))
        return data

    def __len__(self):
        return len(self.data_idx) * self.loop
        # return 1   # debug

    @classmethod
    def _normalize_data_root(cls, data_root):
        data_root = Path(data_root).expanduser()
        if not data_root.is_absolute():
            data_root = (Path.cwd() / data_root).resolve()

        candidates = [data_root]
        if data_root.name == 'S3DIS':
            candidates.append(data_root / 's3disfull')
        elif data_root.name == 'raw':
            candidates.append(data_root.parent)
        elif data_root.name != 's3disfull':
            candidates.append(data_root / 's3disfull')

        for candidate in candidates:
            if (candidate / 'raw').is_dir():
                return candidate

        for candidate in candidates:
            if candidate.name == 's3disfull':
                return candidate

        return candidates[-1]

    @classmethod
    def _download_root(cls, data_root):
        if data_root.name == 's3disfull':
            return data_root.parent
        if data_root.name == 'S3DIS':
            return data_root
        return data_root.parent

    @classmethod
    def _ensure_dataset_available(cls, data_root, download):
        raw_root = data_root / 'raw'
        if raw_root.is_dir():
            return

        if download:
            download_root = cls._download_root(data_root)
            archive_path = download_root / cls.archive_name
            download_root.mkdir(parents=True, exist_ok=True)
            try:
                import gdown
            except ImportError as exc:
                repo_root = Path(__file__).resolve().parents[3]
                venv_python = repo_root / '.venv' / 'bin' / 'python'
                if not venv_python.is_file():
                    raise FileNotFoundError(
                        f'{raw_root} does not exist. Install `gdown` or run `bash script/download_s3dis.sh` '
                        'to prepare the S3DIS dataset.'
                    ) from exc

            if not archive_path.is_file():
                logging.warning(
                    'S3DIS dataset was not found in %s. Downloading it to %s',
                    data_root,
                    archive_path,
                )
                if 'gdown' in locals():
                    gdown.download(cls.download_url, str(archive_path), quiet=False)
                else:
                    subprocess.run(
                        [str(venv_python), '-m', 'gdown', cls.download_url, '-O', str(archive_path)],
                        check=True,
                    )
            else:
                logging.warning(
                    'S3DIS archive already exists at %s. Reusing it to prepare the dataset.',
                    archive_path,
                )

            import tarfile

            with tarfile.open(archive_path) as archive:
                archive.extractall(download_root)

        if not raw_root.is_dir():
            raise FileNotFoundError(
                f'{raw_root} does not exist. Run `bash script/download_s3dis.sh` '
                'or place the extracted dataset under `data/S3DIS/s3disfull/raw`.'
            )


"""debug
from openpoints.dataset import vis_multi_points
import copy
old_data = copy.deepcopy(data)
if self.transform is not None:
    data = self.transform(data)
vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
"""
