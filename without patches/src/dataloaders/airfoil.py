import os
from os.path import exists
import glob
import tarfile
import gdown
from pathlib import Path
from typing import Tuple, Any, List, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset

import scipy.io
import scipy.ndimage
import torchvision
from torchvision import transforms

from src.dataloaders.base import SequenceDataset


def dfp6_norm_fn(data, train=True):
    """Normalizes dfp6 dataset to [0, 1].

    Notes:
        Empirical statistics:
        df6_train_data.max(axis=(0,2,3))
        >>> array([9.98172087e+01, 3.74903321e+01, 1.00000000e+00, 2.91473000e+04,
            1.93697000e+02, 2.20345000e+02])
        df6_train_data.min(axis=(0,2,3))
        >>> array([ 0.00000000e+00, -3.64852235e+01,  0.00000000e+00, -3.97290000e+04,
                -1.56735000e+02, -2.06116000e+02])


    """
    maxes = np.array([
        9.98172087e+01,
        3.74903321e+01,
        1.00000000e+00,
        2.91473000e+04,
        1.93697000e+02,
        2.20345000e+02
        ]
    )[None, :, None, None]
    mins = np.array([
        0.00000000e+00,
        -3.64852235e+01,
        0.00000000e+00,
        -3.97290000e+04,
        -1.56735000e+02,
        -2.06116000e+02
        ]
    )[None, :, None, None]

    data = (data - mins) / (maxes - mins)

    if train:
        err1 = abs(data.max((0,2,3)) -  np.ones(data.shape[1]))
        err0 = abs(data.min((0,2,3)) -  np.zeros(data.shape[1]))
        assert (err1 <= 1e-4).all(), f"{err1}"
        assert (err0 <= 1e-4).all(), f"{err0}"

    return data, (mins, maxes)


def dfpfull_norm_fn(data, train=True):
    """Normalizes dfp dataset to [0, 1].

    Notes:
        Empirical statistics:
        df6_train_data.max(axis=(0,2,3))
        >>> array([9.99706317e+01, 3.78558354e+01, 1.00000000e+00, 2.91473000e+04,│
                1.98526000e+02, 2.20345000e+02])
        df6_train_data.min(axis=(0,2,3))
        >>> array([ 0.0000000e+00, -3.8115759e+01,  0.0000000e+00, -4.0493800e+04,│
                -1.6385200e+02, -2.1577700e+02])

    """
    maxes = np.array([
        9.99706317e+01,
        3.78558354e+01,
        1.00000000e+00,
        2.91473000e+04,
        1.98526000e+02,
        2.20345000e+02
        ]
    )[None, :, None, None]

    mins = np.array([
        0.0000000e+00,
        -3.8115759e+01,
        0.0000000e+00,
        -4.0493800e+04,
       -1.6385200e+02,
       -2.1577700e+02
       ]
    )[None, :, None, None]

    data = (data - mins) / (maxes - mins)

    if train:
        err1 = abs(data.max((0,2,3)) -  np.ones(data.shape[1]))
        err0 = abs(data.min((0,2,3)) -  np.zeros(data.shape[1]))
        assert (err1 <= 1e-4).all(), f"{err1}"
        assert (err0 <= 1e-4).all(), f"{err0}"

    return data, (mins, maxes)


class DFP6Manager:
    id = "1SsC1Fy1ijHzNm0AYsF44K8w1QNOaGLg3"

    def __init__(self, root_dir, download=True):
        self.root_dir = root_dir
        self.train_data_path = os.path.join(root_dir, "data", "train")
        self.test_data_path = os.path.join(root_dir, "data", "test")
        download_exists = not self.download_exists()

        self.to_download = download and download_exists

    def download(self, root_dir):
        os.makedirs(root_dir, exist_ok=True)
        output_path = os.path.join(root_dir, "data.gz")
        gdown.download(id=self.id, quiet=False, output=output_path, resume=True)

    def extract(self, root_dir):

        token_folder_path = os.path.join(root_dir, '*.gz')
        matched_data_path = glob.glob(token_folder_path)
        assert len(matched_data_path) == 1, "Wrong (or no) folder downloaded from GDrive, check ID."

        with tarfile.open(matched_data_path[0]) as tf:
            tf.extractall(path=root_dir, numeric_owner=False)

    def download_exists(self):
        return exists(self.train_data_path) and exists(self.test_data_path)


class DFPFullManager(DFP6Manager):
    id = "1h8ICSxiHr01YV6_cZNKGLdbrzsMvJVlT"


class DFP6Dataset(Dataset):
    """Deep Flow Prediction reduced dataset."""

    def __init__(
        self,
        root_dir,
        split='train',
        valid_frac=0.2,
        transform=None,
        normalize=True,
        f32=True,
    ):

        download_manager = DFP6Manager(root_dir=root_dir, download=True)
        self.download(download_manager, root_dir)
        norm_fn = dfp6_norm_fn if normalize else None
        self.data, self.norm_stats = self.setup(root_dir, split, valid_frac, norm_fn)
        self.transform =  transform
        self.f32 = f32

    def setup(self, root_dir, split, valid_frac, norm_fn=None):
        # Map train / val to train for the purpose of loading data
        path = os.path.join(root_dir, 'data', f'{"train" if split in ["train", "val"] else "test"}')

        path_to_stacked_file = os.path.join(path, 'data_stack.npy')

        # attempt loading of stacked data chunk
        if exists(path_to_stacked_file):
            self.data = np.load(path_to_stacked_file)

        # slow loading of individual data files
        else:
            self.data = []
            if split == "train" or split == "val":
                data_filepaths = os.path.join(path, '*.npz')
                data_filepaths = sorted(glob.glob(data_filepaths))

                for k, path in enumerate(data_filepaths):
                    with np.load(path) as data_file:
                        if type(data_file["a"]) is not None:
                            self.data.append(data_file["a"])
            else:
                data_filepaths = os.path.join(path, '*.npz')
                data_filepaths = sorted(glob.glob(data_filepaths))

                for k, path in enumerate(data_filepaths):
                    with np.load(path) as data_file:
                        if type(data_file["a"]) is not None:
                            self.data.append(data_file["a"])

            # save as stacked for later use
            self.data = np.stack(self.data)
            np.save(path_to_stacked_file, self.data)

        if norm_fn is not None:
            data, norm_stats = norm_fn(self.data, train=split in ['train', 'val'])

        # Split off for validation
        from numpy.random import default_rng
        rng = default_rng(seed=42)
        rng.shuffle(self.data)

        if split == 'train':
            data = data[:int((1 - valid_frac) * data.shape[0])]
        elif split == 'val':
            data = data[int((1 - valid_frac) * data.shape[0]):]
        else:
            pass

        return data, norm_stats

    def download(self, download_manager, root_dir):
      if download_manager.to_download:
            download_manager.download(root_dir)
            download_manager.extract(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        data = torch.tensor(self.data[idx])
        if self.f32: data = data.float()
        if self.transform is not None:
            data = self.transform(data)
        inputs, targets = data[:3], data[3:]
        return inputs, targets


class DFPFullDataset(DFP6Dataset):
    """Deep Flow Prediction full dataset."""

    def __init__(
        self,
        root_dir,
        split='train',
        valid_frac=0.2,
        transform=None,
        normalize=True,
        f32=True,
    ):
        download_manager = DFPFullManager(root_dir=root_dir, download=True)
        self.download(download_manager, root_dir)
        norm_fn = dfpfull_norm_fn if normalize else None
        self.data, self.norm_stats = self.setup(root_dir, split, valid_frac, norm_fn)
        self.transform =  transform
        self.f32 = f32

    def setup(self, root_dir, split, valid_frac, norm_fn=None):
        # Map train / val to train for the purpose of loading data
        path = os.path.join(root_dir, 'data', f'{"train" if split in ["train", "val"] else "test"}')

        path_to_stacked_file =  os.path.join(path, 'data_stack.npy')

        # attempt loading of stacked data chunk
        if exists(path_to_stacked_file):
            self.data = np.load(path_to_stacked_file)

        # slow loading of individual data files
        else:
            self.data = []
            if split == "train" or split == "val":
                for airfoil_class in ["reg", "shear"]:
                    full_path = os.path.join(path, airfoil_class)
                    data_filepaths = os.path.join(full_path, '*.npz')
                    data_filepaths = sorted(glob.glob(data_filepaths))

                    for k, path in enumerate(data_filepaths):
                        with np.load(path) as data_file:
                            if type(data_file["a"]) is not None:
                                self.data.append(data_file["a"])
            else:
                data_filepaths = os.path.join(path, '*.npz')
                data_filepaths = sorted(glob.glob(data_filepaths))

                for k, path in enumerate(data_filepaths):
                    with np.load(path) as data_file:
                        if type(data_file["a"]) is not None:
                            self.data.append(data_file["a"])

            # save as stacked for later use
            self.data = np.stack(self.data)
            np.save(path_to_stacked_file, self.data)

        if norm_fn is not None:
            data, norm_stats = norm_fn(self.data, train=split in ['train', 'val'])

        # Split off for validation
        from numpy.random import default_rng
        rng = default_rng(seed=42)
        rng.shuffle(self.data)

        if split == 'train':
            data = data[:int((1 - valid_frac) * data.shape[0])]
        elif split == 'val':
            data = data[int((1 - valid_frac) * data.shape[0]):]
        else:
            pass

        return data, norm_stats
class DFP(SequenceDataset):
    _name_ = 'dfp'

    @property
    def d_input(self):
        return 6

    @property
    def d_output(self):
        return 6

    @property
    def l_output(self):
        return 6 * self.res ** 2

    @property
    def init_defaults(self):
        return {
            'small': False,
            'res': 128,
            'normalize': True,
            'valid_frac': 0.2,
        }

    def setup(self):
        from src.dataloaders.airfoil import DFP6Dataset, DFPFullDataset
        from torchvision.transforms import InterpolationMode
        self.data_dir = self.data_dir or default_data_path / self._name_ / ('full' if not self.small else 'small')

        transforms = None
        if self.res > 128:
            transforms = torchvision.transforms.Resize(
                (self.res, self.res),
                interpolation=InterpolationMode.BICUBIC,
            )

        cls = DFP6Dataset if self.small else DFPFullDataset

        self.dataset_train = cls(
            self.data_dir,
            split='train',
            valid_frac=self.valid_frac,
            transform=transforms,
        )

        self.dataset_val = cls(
            self.data_dir,
            split='val',
            valid_frac=self.valid_frac,
            transform=transforms,
        )

        self.dataset_test = cls(
            self.data_dir,
            split='test',
            transform=transforms,
        )

