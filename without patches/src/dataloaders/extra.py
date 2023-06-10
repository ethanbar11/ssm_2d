""" Collection of non-core datasets that have been used in the past (e.g. HiPPO paper, rebuttals) or are experimental """

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
# from pytorch_lightning import LightningDataModule

from src.dataloaders.base import default_data_path, SequenceDataset
from .datasets.celeba import _CelebA
from sklearn.model_selection import train_test_split
from src.dataloaders.datasets.impedance import build_gold_clipset, prepare_dataset, build_clipset

class EigenWorms(SequenceDataset):
    _name_ = "eigenworms"
    d_input = 6
    d_output = 5
    l_output = 0
    L = 17984

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        X_train = np.load(self.data_dir / "trainx.npy")
        y_train = np.load(self.data_dir / "trainy.npy")
        X_val   = np.load(self.data_dir / "validx.npy")
        y_val   = np.load(self.data_dir / "validy.npy")
        X_test  = np.load(self.data_dir / "testx.npy")
        y_test  = np.load(self.data_dir / "testy.npy")

        # train, val, test data
        self.dataset_train = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        self.dataset_val = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        self.dataset_test = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    def __str__(self):
        return "eigenworms"

class GestureData(SequenceDataset):
    _name_ = "gesture"
    d_input = 32
    d_output = 5

    training_files = [
        "a3_va3.csv",
        "b1_va3.csv",
        "b3_va3.csv",
        "c1_va3.csv",
        "c3_va3.csv",
        "a2_va3.csv",
        "a1_va3.csv",
    ]

    def setup(self, seq_len=32):
        train_traces = []
        valid_traces = []
        test_traces = []

        interleaved_train = True
        for f in self.dataset_training_files:
            train_traces.extend(
                self.cut_in_sequences(
                    self.load_trace(default_data_path / "ltc/gesture" / f),
                    seq_len,
                    interleaved=interleaved_train,
                )
            )

        train_x, train_y = list(zip(*train_traces))

        self.dataset_train_x = np.stack(train_x, axis=1)
        self.dataset_train_y = np.stack(train_y, axis=1)

        flat_x = self.dataset_train_x.reshape([-1, self.dataset_train_x.shape[-1]])
        mean_x = np.mean(flat_x, axis=0)
        std_x = np.std(flat_x, axis=0)
        self.dataset_train_x = (self.dataset_train_x - mean_x) / std_x

        total_seqs = self.dataset_train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.dataset_train_x = np.swapaxes(self.dataset_train_x, 0, 1)
        self.dataset_train_y = np.swapaxes(self.dataset_train_y, 0, 1)

        self.valid_x = self.dataset_train_x[permutation[:valid_size]]
        self.valid_y = self.dataset_train_y[permutation[:valid_size]]
        self.test_x = self.dataset_train_x[
            permutation[valid_size : valid_size + test_size]
        ]
        self.test_y = self.dataset_train_y[
            permutation[valid_size : valid_size + test_size]
        ]
        self.dataset_train_x = self.dataset_train_x[permutation[valid_size + test_size :]]
        self.dataset_train_y = self.dataset_train_y[permutation[valid_size + test_size :]]

        ## Train data:
        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.dataset_train_x), torch.LongTensor(self.dataset_train_y)
        )

        ## Valid data
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.valid_x), torch.LongTensor(self.valid_y)
        )

        ## Test data
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.test_x), torch.LongTensor(self.test_y)
        )

    def load_trace(self, filename):
        df = pd.read_csv(filename, header=0)

        str_y = df["Phase"].values
        convert = {"D": 0, "P": 1, "S": 2, "H": 3, "R": 4}
        y = np.empty(str_y.shape[0], dtype=np.int32)
        for i in range(str_y.shape[0]):
            y[i] = convert[str_y[i]]

        x = df.values[:, :-1].astype(np.float32)

        return (x, y)

    def cut_in_sequences(self, tup, seq_len, interleaved=False):
        x, y = tup

        num_sequences = x.shape[0] // seq_len
        sequences = []

        for s in range(num_sequences):
            start = seq_len * s
            end = start + seq_len
            sequences.append((x[start:end], y[start:end]))

            if interleaved and s < num_sequences - 1:
                start += seq_len // 2
                end = start + seq_len
                sequences.append((x[start:end], y[start:end]))

        return sequences
    def __str__(self):
        return "gesture"


class CheetahData(SequenceDataset):
    _name_ = "cheetah"
    d_input = 17
    d_output = 17

    @property
    def init_defaults(self):
        return {
            'normalize': False,
            'chunk': True,
        }

    def setup(self, seq_len=32):
        all_files = sorted(
            [
                os.path.join(default_data_path, "ltc/cheetah", d)
                for d in os.listdir(f"{default_data_path}/ltc/cheetah")
                if d.endswith(".npy")
            ]
        )

        train_files = all_files[15:25]
        test_files = all_files[5:15]
        valid_files = all_files[:5]

        self.seq_len = seq_len

        # (length, batch, features)
        self.dataset_train_x, self.dataset_train_y = self._load_files(train_files)
        self.test_x, self.test_y = self._load_files(test_files)
        self.valid_x, self.valid_y = self._load_files(valid_files)

        if self.normalize:
            mean = np.mean(self.dataset_train_x.reshape(-1, 17), axis=0)
            std = np.std(self.dataset_train_x.reshape(-1, 17), axis=0)
            self.dataset_train_x = (self.dataset_train_x - mean) / std
            self.valid_x = (self.valid_x - mean) / std
            self.test_x = (self.test_x - mean) / std

        print("train_x.shape:", str(self.dataset_train_x.shape))
        print("train_y.shape:", str(self.dataset_train_y.shape))
        print("valid_x.shape:", str(self.valid_x.shape))
        print("valid_y.shape:", str(self.valid_y.shape))
        print("test_x.shape:", str(self.test_x.shape))
        print("test_y.shape:", str(self.test_y.shape))

        self.dataset_train_x = np.swapaxes(self.dataset_train_x, 0, 1)
        self.dataset_train_y = np.swapaxes(self.dataset_train_y, 0, 1)
        self.valid_x = np.swapaxes(self.valid_x, 0, 1)
        self.valid_y = np.swapaxes(self.valid_y, 0, 1)
        self.test_x = np.swapaxes(self.test_x, 0, 1)
        self.test_y = np.swapaxes(self.test_y, 0, 1)

        ## Train data:
        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.dataset_train_x), torch.FloatTensor(self.dataset_train_y)
        )

        ## Valid data
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.valid_x), torch.FloatTensor(self.valid_y)
        )

        ## Test data
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.test_x), torch.FloatTensor(self.test_y)
        )

    def _load_files(self, files):
        all_x = []
        all_y = []
        for f in files:

            arr = np.load(f)
            arr = arr.astype(np.float32)

            if self.chunk:
                x, y = self.cut_in_sequences(arr, self.seq_len, 10)

                all_x.extend(x)
                all_y.extend(y)
            else:
                all_x.append(arr[:-1])
                all_y.append(arr[1:])

        return np.stack(all_x, axis=1), np.stack(all_y, axis=1)

    def cut_in_sequences(self, x, seq_len, inc=1):

        sequences_x = []
        sequences_y = []

        for s in range(0, x.shape[0] - seq_len - 1, inc):
            start = s
            end = start + seq_len
            sequences_x.append(x[start:end])
            sequences_y.append(x[start + 1 : end + 1])

        return sequences_x, sequences_y


class Walker2dImitationData(SequenceDataset):
    _name_ = "walker"
    d_input = 18
    d_output = 17

    def setup(self):
        self.seq_len = self.l_output
        all_files = sorted(
            [
                os.path.join(f"{default_data_path}/odelstm/walker", d)
                for d in os.listdir(f"{default_data_path}/odelstm/walker")
                if d.endswith(".npy")
            ]
        )

        self.rng = np.random.RandomState(891374)
        np.random.RandomState(125487).shuffle(all_files)
        # 15% test set, 10% validation set, the rest is for training
        test_n = int(0.15 * len(all_files))
        valid_n = int((0.15 + 0.1) * len(all_files))
        test_files = all_files[:test_n]
        valid_files = all_files[test_n:valid_n]
        train_files = all_files[valid_n:]

        train_x, train_t, train_y = self._load_files(train_files)
        valid_x, valid_t, valid_y = self._load_files(valid_files)
        test_x, test_t, test_y = self._load_files(test_files)

        train_x, train_t, train_y = self.perturb_sequences(
            train_x, train_t, train_y
        )
        valid_x, valid_t, valid_y = self.perturb_sequences(
            valid_x, valid_t, valid_y
        )
        test_x, test_t, test_y = self.perturb_sequences(test_x, test_t, test_y)

        self.dataset_train_x, self.dataset_train_times, self.dataset_train_y = self.align_sequences(
            train_x, train_t, train_y
        )
        self.valid_x, self.valid_times, self.valid_y = self.align_sequences(
            valid_x, valid_t, valid_y
        )
        self.test_x, self.test_times, self.test_y = self.align_sequences(
            test_x, test_t, test_y
        )
        self.d_input = self.dataset_train_x.shape[-1]

        print("train_times: ", str(self.dataset_train_times.shape))
        print("train_x: ", str(self.dataset_train_x.shape))
        print("train_y: ", str(self.dataset_train_y.shape))

        self.dataset_train_x = np.concatenate(
            [self.dataset_train_times, self.dataset_train_x], axis=-1
        )
        self.valid_x = np.concatenate(
            [self.valid_times, self.valid_x], axis=-1
        )
        self.test_x = np.concatenate([self.test_times, self.test_x], axis=-1)

        ## Train data:
        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.dataset_train_x), torch.FloatTensor(self.dataset_train_y)
        )

        ## Valid data
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.valid_x), torch.FloatTensor(self.valid_y)
        )

        ## Test data
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.test_x), torch.FloatTensor(self.test_y)
        )

    def align_sequences(self, set_x, set_t, set_y):

        times = []
        x = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_t = set_t[i]
            seq_y = set_y[i]

            for t in range(
                0, seq_y.shape[0] - self.seq_len, self.seq_len // 4
            ):
                x.append(seq_x[t : t + self.seq_len])
                times.append(seq_t[t : t + self.seq_len])
                y.append(seq_y[t : t + self.seq_len])

        return (
            np.stack(x, axis=0),
            np.expand_dims(np.stack(times, axis=0), axis=-1),
            np.stack(y, axis=0),
        )

    def perturb_sequences(self, set_x, set_t, set_y):

        x = []
        times = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_y = set_y[i]

            new_x, new_times = [], []
            new_y = []

            skip = 0
            for t in range(seq_y.shape[0]):
                skip += 1
                if self.rng.rand() < 0.9:
                    new_x.append(seq_x[t])
                    new_times.append(skip)
                    new_y.append(seq_y[t])
                    skip = 0

            x.append(np.stack(new_x, axis=0))
            times.append(np.stack(new_times, axis=0))
            y.append(np.stack(new_y, axis=0))

        return x, times, y

    def _load_files(self, files):
        all_x = []
        all_t = []
        all_y = []
        for f in files:

            arr = np.load(f)
            x_state = arr[:-1, :].astype(np.float32)
            y = arr[1:, :].astype(np.float32)

            x_times = np.ones(x_state.shape[0])
            all_x.append(x_state)
            all_t.append(x_times)
            all_y.append(y)

            print(
                "Loaded file '{}' of length {:d}".format(f, x_state.shape[0])
            )
        return all_x, all_t, all_y


class ETSMnistData(SequenceDataset):
    _name_ = "etsmnist"
    d_input = 3
    d_output = 10
    l_output = 0

    def setup(self, pad_size=256):
        self.threshold = 128
        self.pad_size = pad_size

        self.data_dir = self.data_dir or default_data_path / "odelstm/etsmnist"
        train_events  = np.load(self.data_dir / "train_events.npy")
        train_elapsed = np.load(self.data_dir / "train_elapsed.npy")
        train_mask    = np.load(self.data_dir / "train_mask.npy")
        train_y       = np.load(self.data_dir / "train_y.npy")

        test_events   = np.load(self.data_dir / "test_events.npy")
        test_elapsed  = np.load(self.data_dir / "test_elapsed.npy")
        test_mask     = np.load(self.data_dir / "test_mask.npy")
        test_y        = np.load(self.data_dir / "test_y.npy")

        print(f"{train_events.shape=}")
        print(f"{train_elapsed.shape=}")
        print(f"{train_mask.shape=}")
        print(f"{train_y.shape=}")

        print(f"{test_events.shape=}")
        print(f"{test_elapsed.shape=}")
        print(f"{test_mask.shape=}")
        print(f"{test_y.shape=}")

        train_elapsed /= self.pad_size
        test_elapsed /= self.pad_size

        train_x = np.concatenate(
            [train_events, train_elapsed, train_mask], axis=-1
        )
        test_x = np.concatenate(
            [test_events, test_elapsed, test_mask], axis=-1
        )

        # This dataset does not come with a validation set
        valid_x = test_x
        valid_y = test_y

        ## Train data:
        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_x), torch.LongTensor(train_y)
        )

        ## Valid data
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(valid_x), torch.LongTensor(valid_y)
        )

        ## Test data
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(test_x), torch.LongTensor(test_y)
        )

class CelebAMultiLabel(SequenceDataset):
    _name_ = "celeba-all"
    d_input = 3
    d_output = 40
    l_output = 0

    @property
    def init_defaults(self):
        return {
            'permute': False,
            'augment': False,
            'res': [178, 218],
            'ndim': 1,
        }

    def setup(self):
        self.L = np.prod(self.res)

        preprocessors = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]

        if self.ndim == 1:
            transform_list = [
                torchvision.transforms.Lambda(
                    lambda x: x.view(self.d_input, self.L).t()
                )  # (L, d_input)
            ]
        else:
            transform_list = []
        if self.permute:
            permutation = permutations.bitreversal_permutation(self.L)
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: x[permutation])
            )

        transforms_train = transform_list
        transforms_eval = (
            [
                torchvision.transforms.CenterCrop(178),
                torchvision.transforms.Resize(self.res),
            ]
            + preprocessors
            + transform_list
        )

        if self.augment:
            augmentations = [
                torchvision.min().RandomResizedCrop(
                    self.res,
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=2,
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
            transforms_train = (
                augmentations + transforms_train
            )

        else:
            transforms_train = (
                [
                    torchvision.transforms.CenterCrop(178),
                    torchvision.transforms.Resize(self.res),
                ]
                + preprocessors
                + transforms_train
            )

        transform_train = torchvision.transforms.Compose(
            transforms_train
        )
        transform_eval = torchvision.transforms.Compose(
            transforms_eval
        )
        self.dataset_train = _CelebA(
            f"{default_data_path}/celeba",
            split="train",
            target_type="attr",
            download=True,
            transform=transform_train,
        )
        self.dataset_val = _CelebA(
            f"{default_data_path}/celeba",
            split="valid",
            target_type="attr",
            transform=transform_eval,
        )
        self.dataset_test = _CelebA(
            f"{default_data_path}/celeba",
            split="test",
            target_type="attr",
            transform=transform_eval,
        )
        self.attr_names = self.dataset_train.attr_names

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class CelebAMultiLabelHQ(CelebAMultiLabel):
    _name_ = "celeba-hq-all"
    @property
    def init_defaults(self):
        return {
            'res': [256, 256],
        }

    def setup(self):
        self.data_dir = f"{default_data_path}/celeba-hq"

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.res),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])
        self.dataset_train = self.dataset_val = None
        self.dataset_test = _CelebA(
            self.data_dir,
            split='hq',
            target_type='attr',
            transform=transform,
        )
        self.attr_names = self.dataset_test.attr_names

    def __str__(self):
        return self._name_


class CelebA(SequenceDataset):
    _name_ = "celeba"
    d_input = 3
    d_output = 2
    l_output = 0

    @property
    def init_defaults(self):
        return {
            'permute': False,
            'augment': False,
            'target': 'Wearing_Lipstick',
        }

    def setup(self):
        preprocessors = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]

        transform_list = [
            torchvision.transforms.Lambda(
                lambda x: x.view(self.d_input, self.L).t()
            )  # (L, d_input)
        ]
        if self.permute:
            permutation = permutations.bitreversal_permutation(self.L)
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: x[permutation])
            )

        torchvision.transforms_train = transform_list
        torchvision.transforms_eval = (
            [
                torchvision.transforms.CenterCrop(178),
                torchvision.transforms.Resize((178, 218)),
            ]
            + preprocessors
            + transform_list
        )

        # if "augment" in self.dataset_cfg and self.dataset_cfg.augment:
        if self.augment:
            augmentations = [
                torchvision.transforms.RandomResizedCrop(
                    (178, 218),
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=2,
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
            torchvision.transforms_train = (
                augmentations + torchvision.transforms_train
            )

        else:
            torchvision.transforms_train = (
                [
                    torchvision.transforms.CenterCrop(178),
                    torchvision.transforms.Resize((178, 218)),
                ]
                + preprocessors
                + torchvision.transforms_train
            )

        transform_train = torchvision.transforms.Compose(
            torchvision.transforms_train
        )
        transform_eval = torchvision.transforms.Compose(
            torchvision.transforms_eval
        )
        self.dataset_train = _CelebA(
            f"{default_data_path}/{self._name_}",
            task=self.target,
            split="train",
            target_type="attr",
            download=True,
            transform=transform_train,
        )
        self.dataset_val = _CelebA(
            f"{default_data_path}/{self._name_}",
            task=self.target,
            split="valid",
            target_type="attr",
            transform=transform_eval,
        )
        self.dataset_test = _CelebA(
            f"{default_data_path}/{self._name_}",
            task=self.target,
            split="test",
            target_type="attr",
            transform=transform_eval,
        )

    def __str__(self):
        return f"{'p' if self.dataset_cfg.permute else 's'}{self._name_}"

class CharacterTrajectories(SequenceDataset):
    """CharacterTrajectories dataset from the UCI Machine Learning archive.

    See datasets.uea.postprocess_data for dataset configuration settings.
    """

    _name_ = "ct"
    d_input = 3
    d_output = 20
    l_output = 0

    @property
    def init_defaults(self):
        return {
            # Sampling frequency relative to original sequence
            'train_hz': 1,
            'eval_hz': 1,
            # Sample evenly or randomly
            'train_uniform': True,
            'eval_uniform': True,
            # Include timestamps in input
            'timestamp': False,
            # Timestamp scale (multiplier on timestamp)
            'train_ts': 1,
            'eval_ts': 1,
        }

    def setup(self):
        if self.timestamp or self.mask:
            self.d_input += 1
        from src.dataloaders import uea

        *data, num_classes, input_channels = uea.get_data(
            "CharacterTrajectories",
            intensity=False,
        )
        self.dataset_train, self.dataset_val, self.dataset_test = uea.postprocess_data(
            *data,
            train_hz=self.dataset_train_hz,
            eval_hz=self.eval_hz,
            train_uniform=self.dataset_train_uniform,
            eval_uniform=self.eval_uniform,
            timestamp=self.timestamp,
            train_ts=self.dataset_train_ts,
            eval_ts=self.eval_ts,
            mask=self.mask,
        )
        assert (
            num_classes == self.d_output
        ), f"Output size should be {num_classes}"

class Markov(SequenceDataset):
    _name_ = "markov"

    @property
    def n_tokens(self): return self.n_alphabet+1

    @property
    def init_defaults(self):
        return {
            'l_seq': 100, # length of sequence
            'n_alphabet': 3,  # alphabet size # TODO pass into constructor
            'n_samples': 50000,
            'val_split': 0.1,
        }

    def setup(self):
        from .datasets.markov import torch_markov_data
        targets = torch_markov_data(
            self.n_samples,
            self.l_seq,
        )
        inputs = targets.roll(1, dims=-1)
        inputs[:, 0] = self.n_alphabet

        self.dataset_train = torch.utils.data.TensorDataset(inputs, targets)

        self.split_train_val(self.val_split)
        self.dataset_test = None

    def __str__(self):
        return f"{self._name_}{self.l_noise}{'v' if self.variable else ''}"


class Integrator(SequenceDataset):
    _name_ = "integrator"

    @property
    def d_input(self): return 1

    @property
    def d_output(self): return 1

    @property
    def init_defaults(self):
        return {
            'l_seq': 1024, # length of sequence
            'n_components': 10, # number of sins to mix
            'max_ampl': 10.0,
            'max_freq': 100.0,
            'n_samples': 100000,
            'val_split': 0.1,
        }

    def setup(self):
        from .datasets.integrator import integrator_data
        data, targets = integrator_data(
            self.n_samples,
            self.l_seq,
            self.n_components,
            self.max_ampl,
            self.max_freq,
        )
        self.dataset_train = torch.utils.data.TensorDataset(data.unsqueeze(-1), targets.unsqueeze(-1))

        self.split_train_val(self.val_split)
        self.dataset_test = None

        self.collate_fn = None

    def __str__(self):
        return f"{self._name_}"


class Impedance(SequenceDataset):
    _name_ = "impedance"

    # train_study_ids, dev_study_ids = train_test_split([1, 3, 4, 5, 7, 8, 11, 12, 17, 18, 19, 20, 22, 23, 24], test_size=0.2, random_state=42) # train-dev split
    train_study_ids, dev_study_ids = train_test_split([1, 2, 3, 4, 5, 7, 8, 11, 12, 17, 18, 19, 20, 22, 23, 24, 26, 29, 32, 33, 48, 49, 50, 51, 52], test_size=0.2, random_state=42) # train-dev split
    train_study_ids, dev_study_ids = sorted(train_study_ids), sorted(dev_study_ids)
    # test_study_ids = [26, 29, 32, 33]
    # test_study_ids = [34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
    test_study_ids = [34, 38, 39, 40, 41, 42, 43, 44, 45, 46] # 47 # [48, 49, 50, 51, 52]
    # test_study_ids_v2 = [48, 49, 50, 51, 52] # these are additional studies that may need to be moved into training
    test_study_ids_v3 = [147, 148, 149, 150, 151, 152, 153, 154, 155, 156] # previously named [47, 48, 49, 50, 51, 52, 53, 54, 55, 56]

    # train_study_ids = train_study_ids + test_study_ids_v2
    # Not OK [48, 49, 50, 51, 52] but can use truth/48.txt, ... instead of truth/gold.csv
    # ^ the values in truth/gold.csv are not the same as the values in the .txt files
    # Not OK [53, 54, 55, 56] -- test AUCs don't track val at all (most likely the .txt files in data/ are wrong)
    # OK [34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

    d_output = 2
    l_output = 0

    @property
    def init_defaults(self):
        return {
            'studies': [1, 2, 3, 4, 5, 7, 8, 11, 12, 17, 18, 19, 23, 32],
            'build_clip_duration': 300, # seconds
            'sensors': [0, 1, 2, 3, 4, 5],
            'scaling': 'none', # ['none', 'standardize', 'minmax']
            'val_size': 0.2,
            'test_variant': 0, # [0: calculate AUCs as if parsing entire studies, 1: calculate AUCs only on events / non-events in gold reads]
            # 0 will yield higher AUCs since the set of non-events will reflect the real distribution
            # 1 will yield lower AUCs since the set of non-events will be harder (only those that were considered events by human or existing algorithm)
        }

    @property
    def d_input(self):
        return len(self.sensors)

    def init(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

    def setup(self):
        train_clipset = build_clipset(str(self.data_dir), self.train_study_ids, self.build_clip_duration * 1000)
        dev_clipset = build_clipset(str(self.data_dir), self.dev_study_ids, self.build_clip_duration * 1000)
        # # test_clipset = build_clipset(str(self.data_dir), self.test_study_ids, self.build_clip_duration * 1000)

        # test_clipset = build_gold_clipset(str(self.data_dir), self.test_study_ids, self.build_clip_duration * 1000, variant=0)
        # test_clipset_2 = build_clipset(str(self.data_dir), self.test_study_ids_v2, self.build_clip_duration * 1000)
        # test_clipset = np.concatenate([test_clipset[0], test_clipset_2[0]], axis=0), np.concatenate([test_clipset[1], test_clipset_2[1]], axis=0)
        # test_clipset = build_gold_clipset(str(self.data_dir), self.test_study_ids, self.build_clip_duration * 1000, variant=1)

        self.dev_clipsets_by_study = []
        for study_id in self.dev_study_ids:
            self.dev_clipsets_by_study.append(build_clipset(str(self.data_dir), [study_id], self.build_clip_duration * 1000))

        self.test_clipsets_by_study_0 = []
        for study_id in self.test_study_ids + self.test_study_ids_v3:
            self.test_clipsets_by_study_0.append(build_gold_clipset(str(self.data_dir), [study_id], self.build_clip_duration * 1000, variant=0))

        test_clipset_0 = np.concatenate([e[0] for e in self.test_clipsets_by_study_0], axis=0), np.concatenate([e[1] for e in self.test_clipsets_by_study_0], axis=0)

        self.test_clipsets_by_study_1 = []
        for study_id in self.test_study_ids + self.test_study_ids_v3:
            self.test_clipsets_by_study_1.append(build_gold_clipset(str(self.data_dir), [study_id], self.build_clip_duration * 1000, variant=1))

        test_clipset_1 = np.concatenate([e[0] for e in self.test_clipsets_by_study_1], axis=0), np.concatenate([e[1] for e in self.test_clipsets_by_study_1], axis=0)

        # test_clipset = build_gold_clipset(str(self.data_dir), self.test_study_ids, self.build_clip_duration * 1000, variant=self.test_variant)
        # test_clipset_2 = build_gold_clipset(str(self.data_dir), self.test_study_ids_v3, self.build_clip_duration * 1000, variant=self.test_variant)
        # test_clipset = np.concatenate([test_clipset[0], test_clipset_2[0]], axis=0), np.concatenate([test_clipset[1], test_clipset_2[1]], axis=0)


        train_X, _, train_y, _ = prepare_dataset(
            train_clipset,
            order='NCHW',
            y_format='classindex',
            dev_size=1e-5,
            scaling=self.scaling,
        )

        dev_X, _, dev_y, _ = prepare_dataset(
            dev_clipset,
            order='NCHW',
            y_format='classindex',
            dev_size=1e-5,
            scaling=self.scaling,
        )

        test_X_0, _, test_y_0, _ = prepare_dataset(
            test_clipset_0,
            order='NCHW',
            y_format='classindex',
            dev_size=1e-5,
            scaling=self.scaling,
        )

        test_X_1, _, test_y_1, _ = prepare_dataset(
            test_clipset_1,
            order='NCHW',
            y_format='classindex',
            dev_size=1e-5,
            scaling=self.scaling,
        )

        train_X = torch.tensor(train_X.squeeze())
        dev_X = torch.tensor(dev_X.squeeze())
        test_X_0 = torch.tensor(test_X_0.squeeze())
        test_X_1 = torch.tensor(test_X_1.squeeze())

        # Balance the data
        n_1 = train_y[train_y == 1].shape[0]
        train_X_balanced = np.concatenate([train_X[train_y == 0][:n_1], train_X[train_y == 1]], axis=0)
        train_y_balanced = np.concatenate([train_y[train_y == 0][:n_1], train_y[train_y == 1]], axis=0)

        self.dataset_train = torch.utils.data.TensorDataset(torch.tensor(train_X_balanced).to(torch.float), torch.tensor(train_y_balanced))
        self.dataset_val = torch.utils.data.TensorDataset(torch.tensor(dev_X).to(torch.float), torch.tensor(dev_y))
        self.dataset_test_0 = torch.utils.data.TensorDataset(torch.tensor(test_X_0).to(torch.float), torch.tensor(test_y_0))
        self.dataset_test_1 = torch.utils.data.TensorDataset(torch.tensor(test_X_1).to(torch.float), torch.tensor(test_y_1))

        self.val_pos = torch.utils.data.TensorDataset(torch.tensor(dev_X[dev_y == 1]).to(torch.float), torch.tensor(dev_y[dev_y == 1]))
        self.val_neg = torch.utils.data.TensorDataset(torch.tensor(dev_X[dev_y == 0]).to(torch.float), torch.tensor(dev_y[dev_y == 0]))
        self.test_pos_0 = torch.utils.data.TensorDataset(torch.tensor(test_X_0[test_y_0 == 1]).to(torch.float), torch.tensor(test_y_0[test_y_0 == 1]))
        self.test_neg_0 = torch.utils.data.TensorDataset(torch.tensor(test_X_0[test_y_0 == 0]).to(torch.float), torch.tensor(test_y_0[test_y_0 == 0]))
        self.test_pos_1 = torch.utils.data.TensorDataset(torch.tensor(test_X_1[test_y_1 == 1]).to(torch.float), torch.tensor(test_y_1[test_y_1 == 1]))
        self.test_neg_1 = torch.utils.data.TensorDataset(torch.tensor(test_X_1[test_y_1 == 0]).to(torch.float), torch.tensor(test_y_1[test_y_1 == 0]))

        print("Validation Dataset Sizes:")
        print(len(self.dataset_val), len(self.val_pos), len(self.val_neg))
        print("Test Dataset_0 Sizes:")
        print(len(self.dataset_test_0), len(self.test_pos_0), len(self.test_neg_0))
        print("Test Dataset_1 Sizes:")
        print(len(self.dataset_test_1), len(self.test_pos_1), len(self.test_neg_1))

    def val_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            collate_fn=None,
            **kwargs,
        )

    def test_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        return {
            'gold': torch.utils.data.DataLoader(
                self.dataset_test_1,
                collate_fn=None,
                **kwargs,
            ),
            'gold+nonevents': torch.utils.data.DataLoader(
                self.dataset_test_0,
                collate_fn=None,
                **kwargs,
            ),
        }


class MemoryCopying(SequenceDataset):
    _name_ = "context"

    @property
    def init_defaults(self):
        return {
            'batch_size': 32,
            'l_max': 32,
            'n_tokens': 8,  # alphabet size
            'n_delay': 1,
            'n_samples': 65536,
        }

    @property
    def d_output(self): return self.n_tokens

    def setup(self):
        tokens = torch.randint(low=0, high=self.n_tokens-1, size=(self.n_samples,))
        x = rearrange(tokens, '(n b l) -> n b l', b=self.batch_size, l=self.l_max)
        y = F.pad(x[:-self.n_delay], (0, 0, 0, 0, self.n_delay, 0))
        x = rearrange(x, 'n b l -> (n b) l')
        y = rearrange(y, 'n b l -> (n b) l')
        x = x.unsqueeze(-1)
        self.dataset_train = torch.utils.data.TensorDataset(x, y)
        self.dataset_val = None
        self.dataset_test = None

    def __str__(self):
        return f"{self._name_}{self.l_max}{self.n_delay}"

    def train_dataloader(self, **kwargs):
        return torch.utils.data.DataLoader(self.dataset_train, num_workers=1, shuffle=False, batch_size=self.batch_size)

class EMA(SequenceDataset):
    _name_ = "ema"

    @property
    def d_input(self): return 1

    @property
    def d_output(self): return 1

    @property
    def init_defaults(self):
        return {
            'l_seq': 1024, # length of sequence
            'n_components': 10, # number of sins to mix
            'max_ampl': 10.0,
            'max_freq': 100.0,
            'horizon': 1024,
            'ema': True,
            'n_samples': 100000,
            'val_split': 0.1,
        }

    def setup(self):
        from .datasets.integrator import integrator_data
        from src.models.functional.toeplitz import causal_convolution

        data, _ = integrator_data(
            self.n_samples,
            self.l_seq,
            self.n_components,
            self.max_ampl,
            self.max_freq,
        ) # (B, L)
        if self.ema:
            weight = (1/self.horizon) * (1.-1/self.horizon) ** torch.arange(data.size(-1))
            targets = causal_convolution(data, weight)
        else:
            cs = torch.cumsum(data, dim=-1)
            delay = F.pad(cs, (2*self.horizon, 0))[:, :cs.size(-1)]
            targets = (cs - delay) / (2 * self.horizon)
        Z = torch.mean(targets**2)
        targets = targets / torch.sqrt(Z)
        self.dataset_train = torch.utils.data.TensorDataset(data.unsqueeze(-1), targets.unsqueeze(-1))

        self.split_train_val(self.val_split)
        self.dataset_test = None

        self.collate_fn = None

    def __str__(self):
        return f"{self._name_}"

class SMA(SequenceDataset):
    _name_ = "sma"

    @property
    def init_defaults(self):
        return {
            "l_seq": 1024, # length of total sequence
            "l_mem": 256,
            "dt": 0.001,
            "freq": 100.0,
            "static": False, # Use a static dataset of size n_train, otherwise always use random data with n_train per epoch
            "n_train": 10000,
            "n_eval": 1000,
            # "val_split": 0.1,
        }

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 1

    @property
    def l_output(self):
        return self.l_seq

    def setup(self):
        from .datasets.sma import SMATrainDataset, SMAEvalDataset

        if self.static: train_cls = SMAEvalDataset
        else: train_cls = SMATrainDataset

        self.dataset_train = train_cls(
            samples=self.n_train,
            l_seq=self.l_seq,
            l_mem=self.l_mem,
            dt=self.dt,
            freq=self.freq,
        )
        self.dataset_val = SMAEvalDataset(
            samples=self.n_eval,
            l_seq=self.l_seq,
            l_mem=self.l_mem,
            dt=self.dt,
            freq=self.freq,
        )
        self.dataset_test = None


    def __str__(self):
        raise NotImplementedError
class STFT(SequenceDataset):
    _name_ = "stft"

    @property
    def init_defaults(self):
        return {
            "l_seq": 1024, # length of total sequence
            "l_fft": 256,
            "dt": 0.001,
            "freq": 100.0,
            "static": False, # Use a static dataset of size n_train, otherwise always use random data with n_train per epoch
            "n_train": 10000,
            "n_eval": 1000,
            # "val_split": 0.1,
        }

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return self.l_fft

    @property
    def l_output(self):
        return self.l_seq

    def setup(self):
        from .datasets.stft import STFTTrainDataset, STFTEvalDataset

        if self.static: train_cls = STFTEvalDataset
        else: train_cls = STFTTrainDataset

        self.dataset_train = train_cls(
            samples=self.n_train,
            l_seq=self.l_seq,
            l_fft=self.l_fft,
            dt=self.dt,
            freq=self.freq,
        )
        self.dataset_val = STFTEvalDataset(
            samples=self.n_eval,
            l_seq=self.l_seq,
            l_fft=self.l_fft,
            dt=self.dt,
            freq=self.freq,
        )
        self.dataset_test = None


    def __str__(self):
        raise NotImplementedError
