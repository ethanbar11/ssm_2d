import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dataloaders.utils.signal import whitesignal


class SMATrainDataset(torch.utils.data.Dataset):
    def __init__(self, samples, l_seq=1024, l_mem=256, dt=1e-3, freq=1.0):
        """
        """
        super().__init__()
        self.L = l_seq
        self.l_mem = l_mem
        self.dt = dt
        self.freq = freq
        self.samples = samples

    def __getitem__(self, idx):
        assert 0 <= idx < self.samples
        x = whitesignal(self.L*self.dt, self.dt, self.freq) # (l_seq)
        x = torch.FloatTensor(x)
        cx = torch.cumsum(x, dim=-1)
        delay = F.pad(cx, (self.l_mem, 0))[..., :cx.size(-1)]
        y = (cx - delay) # / self.l_mem
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        return x, y

    def __len__(self):
        return self.samples


class SMAEvalDataset(torch.utils.data.TensorDataset):
    def __init__(self, samples, l_seq=1024, l_mem=256, dt=1e-3, freq=1.0):
        # self.T = T
        self.L = l_seq
        self.l_mem = l_mem
        self.dt = dt
        self.freq = freq
        self.samples = samples

        X = whitesignal(self.L*self.dt, self.dt, self.freq, batch_shape=(self.samples,))
        X = torch.FloatTensor(X)
        cx = torch.cumsum(X, dim=-1)
        delay = F.pad(cx, (self.l_mem, 0))[..., :cx.size(-1)]
        Y = (cx - delay) # / self.l_mem
        X = X.unsqueeze(-1) # (batch, l_seq, 1)
        Y = Y.unsqueeze(-1)

        super().__init__(X, Y)



if __name__ == '__main__':
    # a = torch_copying_data(20, 5, 10, batch_shape=(3,))
    # print(a)

    print("SMA Train Dataset")
    ds = SMATrainDataset(samples=100)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
    it = iter(loader)
    print(next(it))
    # for (x, y) in enumerate(loader):
    #     print(x, y)

    print("SMA Evaluation Dataset")
    eval_ds = SMAEvalDataset(samples=5)
    loader = torch.utils.data.DataLoader(eval_ds, batch_size=2, num_workers=2)
    for (x, y) in loader:
        print(x, y)

