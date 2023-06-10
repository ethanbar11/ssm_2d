import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dataloaders.utils.signal import whitesignal


class DelayTrainDataset(torch.utils.data.Dataset):
    def __init__(self, samples, l_seq=1024, n_lag=1, l_lag=None, dt=1e-3, freq=1.0):
        """
        """
        super().__init__()
        self.L = l_seq
        self.dt = dt
        self.freq = freq
        self.samples = samples
        self.l_lag = l_lag or l_seq // n_lag
        self.n_lag = n_lag

    def __getitem__(self, idx):
        assert 0 <= idx < self.samples
        x = torch.FloatTensor(whitesignal(self.L*self.dt, self.dt, self.freq)) # (l_seq)
        y = torch.stack([
            F.pad(x[:self.L-i*self.l_lag], (i*self.l_lag, 0))
            for i in range(1, self.n_lag+1)
        ], dim=-1) # (l_seq, n_lag)
        x = x.unsqueeze(-1)
        return x, y

    def __len__(self):
        return self.samples


class DelayEvalDataset(torch.utils.data.TensorDataset):
    def __init__(self, samples, l_seq=1024, n_lag=1, l_lag=None, dt=1e-3, freq=1.0):
        self.L = l_seq
        self.dt = dt
        self.freq = freq
        self.samples = samples
        self.l_lag = l_lag or l_seq // n_lag
        self.n_lag = n_lag

        X = torch.FloatTensor(whitesignal(self.L*self.dt, self.dt, self.freq, batch_shape=(self.samples,))) # (samples, l_seq, 1)
        Y = torch.stack([
            F.pad(X[:, :self.L-i*self.l_lag], (i*self.l_lag, 0)) # manually subtract from self.L otherwise error in i=0 case
            for i in range(1, self.n_lag+1)
        ], dim=-1) # (batch, l_seq, n_lag)
        X = X.unsqueeze(-1) # (batch, l_seq, 1)

        super().__init__(X, Y)



if __name__ == '__main__':
    print("Delay Train Dataset")
    ds = DelayTrainDataset(samples=100, l_seq=16, n_lag=1, l_lag=4, dt=0.5)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
    it = iter(loader)
    x, y = next(it)
    print(x.shape, y.shape)
    print("x:", x[0, :, 0])
    print("y:", y[0, :, 0])
    # for (x, y) in enumerate(loader):
    #     print(x, y)

    print("Delay Evaluation Dataset")
    eval_ds = DelayEvalDataset(samples=5)
    loader = torch.utils.data.DataLoader(eval_ds, batch_size=2, num_workers=2)
    for (x, y) in loader:
        print(x, y)

