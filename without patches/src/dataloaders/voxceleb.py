import torchaudio
import pandas as pd
import os
import torch
import math
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from src.dataloaders.base import SequenceDataset, default_data_path
from src.dataloaders.esc import ESC50Dataset

def random_clip(x, freq, seconds):
    """
    Construct a random clip of length seconds from the signal x sampled at freq.
    """
    start = torch.randint(0, x.size(0) - 1 - seconds * freq, (1,))[0]
    end = start + seconds * freq
    return x[start:end]

def rms_energy(x):
    return 10 * np.log10((1e-12 + x.dot(x))/len(x))


def sox_loader(path, sr, effects=False, resample=None):
    if effects:
        # Define effects
        effects = [
            # ["lowpass", "-1", "300"], # apply single-pole lowpass filter
            ["speed", f"{np.random.uniform(0.8, 1.2)}"], # adjust speed
            ["rate", f"{sr}"], # resample to 16000Hz
        ]

        # if np.random.rand() < 0.5:
        #     effects.append(["reverb", "-w"]) # add reverb

    else:
        effects = []

    if resample is not None:
        effects.extend([
            ["lowpass", f"{resample // 2}"],
            ["rate", f'{resample}'],
        ])

    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

class _VoxCeleb(Dataset):
    """
    VoxCeleb sampling rate is 16000Hz:
    Example output of torchaudio.info on a wav file in the dataset
        AudioMetaData(sample_rate=16000, num_frames=103041, num_channels=1, bits_per_sample=16, encoding=PCM_S)
    """
    FREQ = 16000
    N_CLASSES=1251

    def __init__(self, data_dir, split, clip_length=3, effects=False, noise_dir=None, num_classes=False, self_normalize=False):
        info = pd.read_csv(os.path.join(data_dir, 'iden_split.txt'), sep=" ", header=None, names=['split', 'path'])
        info['id'] = info['path'].str.split("/", n=1, expand=True)[0].str.replace("id", "").astype(int)
        info['label'] = (info['id'].values - 10001).astype(int)
        info['path'] = "wav/" + info['path']

        self.metadata = pd.read_csv(os.path.join(data_dir, 'vox1_meta.csv'), sep='\t')

        self.splits = info[info['split'] == {'train': 1, 'val': 2, 'test': 3}[split]]

        if num_classes is not None and isinstance(num_classes, int):
            self.splits = self.splits[self.splits['label'].isin(list(range(num_classes)))]
            self.N_CLASSES = num_classes

        self.X = self.splits['path'].values
        self.y = self.splits['label'].values

        self.clip_length = clip_length

        self.data_dir = data_dir

        self.noise = self._load_noise(noise_dir)
        self.effects = effects
        self.self_normalize = self_normalize

    def _load_noise(self, noise_dir):
        if noise_dir is None:
            return None
        return ESC50Dataset(noise_dir, None, self.clip_length)

    def __len__(self):
        return len(self.y)

    def add_noise(self, audio):
        if self.noise is not None:
            # Sample a random noise clip (remove the label)
            noise = self.noise.random()[0]

            # audio_power = audio.norm(p=2)
            # noise_power = noise.norm(p=2)
            # snr_db = np.random.choice([3, 10, 20])
            # snr = math.exp(snr_db / 10)
            # scale = snr * noise_power / audio_power

            audio_dB = rms_energy(audio.squeeze())
            noise_dB = rms_energy(noise.squeeze())

            snr = np.random.uniform(low=-20.0, high=60.0)
            noise_target_dB = audio_dB - snr

            noise_scaled = 10**(noise_target_dB/20) * noise / 10**(noise_dB/20)

            audio = audio + noise_scaled

        return audio

    # def add_effects(self, audio, sr):
    #     if not self.effects:
    #         return audio

    #     # Define effects
    #     effects = [
    #         # ["lowpass", "-1", "300"], # apply single-pole lowpass filter
    #         ["speed", f"{np.random.uniform(0.8, 1.2)}"], # adjust speed
    #         ["rate", f"{sr}"], # resample to 16000Hz
    #     ]

    #     if np.random.rand() < 0.5:
    #         effects.append(["reverb", "-w"]) # add reverb

    #     # Apply effects
    #     audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sr, effects)

    #     return audio


    def __getitem__(self, idx):
        label = self.y[idx]
        # audio, sr = torchaudio.load(os.path.join(self.data_dir,self.X[idx]))
        if self.effects:
            audio, sr = sox_loader(os.path.join(self.data_dir, self.X[idx]), sr=self.FREQ, effects=self.effects)
        else:
            audio, sr = torchaudio.load(os.path.join(self.data_dir, self.X[idx]))

        audio = audio.T
        audio = random_clip(audio, self.FREQ, self.clip_length)
        audio = self.add_noise(audio)
        if self.self_normalize:
            audio = (audio - audio.mean()) / audio.std()
        # audio = self.add_effects(audio, sr)

        # Audio has shape (FREQ * num_seconds, 1)
        return audio, label

class VoxCeleb(SequenceDataset):
    _name_ = "voxceleb"

    l_output = 0
    d_input = 1

    init_defaults = {
        'clip_length': 3,
        'num_classes': None,
        'noise': False,
        'effects': False,
        'self_normalize': False,
    }

    @property
    def d_output(self):
        return _VoxCeleb.N_CLASSES if self.num_classes is None else self.num_classes

    def init(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.noise_dir = default_data_path / 'ESC-50'

    def setup(self):
        self.dataset_train = _VoxCeleb(self.data_dir, 'train', self.clip_length, self.effects, self.noise_dir if self.noise else None, self.num_classes, self_normalize=self.self_normalize)
        # self.split_train_val(0.5)
        self.dataset_val = _VoxCeleb(self.data_dir, 'val', self.clip_length, num_classes=self.num_classes, self_normalize=self.self_normalize)
        self.dataset_test = _VoxCeleb(self.data_dir, 'test', self.clip_length, num_classes=self.num_classes, self_normalize=self.self_normalize)

        self.collate_fn = None

