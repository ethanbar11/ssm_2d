import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.dataloaders.utils.signal import whitesignal


def stft(x, l_fft):
    # x: (..., L)
    assert l_fft % 2 == 0
    l_seq = x.shape[-1]
    x_pad = np.concatenate([np.zeros(x.shape[:-1]+(l_fft-1,)), x], axis=-1)
    idx = np.arange(l_fft)[None, :] + np.arange(l_seq)[:, None] # (l_seq, l_fft)
    x_windows = x_pad[..., idx] # (..., l_seq, l_fft)
    x_stft = np.fft.rfft(x_windows) # (..., l_seq, l_fft/2+1)
    x_stft = np.concatenate([x_stft[..., :-1].real, x_stft[..., :-1].imag], axis=-1) # (..., l_seq, l_fft)
    return x_stft

class STFTTrainDataset(torch.utils.data.Dataset):
    def __init__(self, samples, l_seq=1024, l_fft=256, dt=1e-3, freq=1.0):
        """
        """
        super().__init__()
        self.L = l_seq
        self.l_fft = l_fft
        self.dt = dt
        self.freq = freq
        self.samples = samples


    def __getitem__(self, idx):
        assert 0 <= idx < self.samples
        x = whitesignal(self.L*self.dt, self.dt, self.freq) # (l_seq)
        y = stft(x, self.l_fft) # (l_seq, l_fft)
        x = torch.FloatTensor(x).unsqueeze(-1)
        y = torch.FloatTensor(y)
        return x, y

    def __len__(self):
        return self.samples


class STFTEvalDataset(torch.utils.data.TensorDataset):
    def __init__(self, samples, l_seq=1024, l_fft=256, dt=1e-3, freq=1.0):
        self.L = l_seq
        self.l_fft = l_fft
        self.dt = dt
        self.freq = freq
        self.samples = samples

        X = whitesignal(self.L*self.dt, self.dt, self.freq, batch_shape=(self.samples,))
        Y = stft(X, self.l_fft) # (batch, l_seq, l_fft)
        Y = torch.FloatTensor(Y)
        X = torch.FloatTensor(X).unsqueeze(-1) # (batch, l_seq, 1)
        print("shapes", X.shape, Y.shape)

        super().__init__(X, Y)



if __name__ == '__main__':
    # a = torch_copying_data(20, 5, 10, batch_shape=(3,))
    # print(a)

    print("STFT Train Dataset")
    ds = STFTTrainDataset(samples=100)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
    it = iter(loader)
    print(next(it))
    # for (x, y) in enumerate(loader):
    #     print(x, y)

    print("STFT Evaluation Dataset")
    eval_ds = STFTEvalDataset(samples=5)
    loader = torch.utils.data.DataLoader(eval_ds, batch_size=2, num_workers=2)
    for (x, y) in loader:
        print(x, y)

