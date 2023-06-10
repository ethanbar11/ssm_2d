import torchaudio
import pandas as pd
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import glob

from src.dataloaders.base import SequenceDataset, default_data_path

def random_clip(x, freq, seconds):
    """
    Construct a random clip of length seconds from the signal x sampled at freq.
    """
    start = torch.randint(0, x.size(0) - 1 - seconds * freq, (1,))[0]
    end = start + seconds * freq
    return x[start:end]

def sox_loader(path, effects=None, resample=None):

    if effects is None:
        effects = [
            ["remix", "1"]
        ]

    if resample is not None:
        effects.extend([
            ["lowpass", f"{resample // 2}"],
            ["rate", f'{resample}'],
        ])

    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


class ESC50Dataset(Dataset):
    """
    ESC-50 sampling rate is 44100Hz
    """
    FREQ = 44100
    N_CLASSES=50

    def __init__(self, data_dir, split, clip_length=3, sampling_rate=16000,):

        self.data_dir = data_dir
        self.X, self.y, self.folds = self._load_files(resample=sampling_rate)
        self.X = self.X.transpose(1, 2)

        # self.splits = info[info['split'] == {'train': 1, 'val': 2, 'test': 3}[split]]
        self.split = split
        self.clip_length = clip_length

        self.FREQ = sampling_rate

    def _load_files(self, effects=None, resample=None):
        X = []
        y = []
        folds = []
        for wav in glob.glob(os.path.join(self.data_dir, "audio", "*.wav")):
            audio, sr = sox_loader(wav, effects=effects, resample=resample)
            X.append(audio)
            y.append(int(wav.split("/")[-1].split(".")[0].split("-")[-1]))
            folds.append(int(wav.split("/")[-1].split(".")[0].split("-")[0]))

        return torch.stack(X), torch.tensor(y), torch.tensor(folds)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label = self.y[idx]
        audio = self.X[idx]

        # Get a random clip
        audio = random_clip(audio, self.FREQ, self.clip_length)

        # Audio has shape (FREQ * num_seconds, 1)
        return audio, label

    def random(self):
        idx = np.random.randint(0, len(self))
        return self[idx]

# class ESC50(SequenceDataset):
#     _name_ = "esc50"

#     d_output = _ESC50.N_CLASSES
#     l_output = 0
#     d_input = 1

#     init_defaults = {
#         'clip_length': 3,
#     }

#     def init(self):
#         self.data_dir = self.data_dir or default_data_path / self._name_

#     def setup(self):

#         self.train = _ESC50(self.data_dir, 'train', self.clip_length)
#         # self.split_train_val(0.5)
#         self.val = _ESC50(self.data_dir, 'val', self.clip_length)
#         self.test = _ESC50(self.data_dir, 'test', self.clip_length)

#         self.collate_fn = None

