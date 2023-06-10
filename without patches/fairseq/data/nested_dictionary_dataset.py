# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from einops import rearrange
from collections import OrderedDict

import torch
import torchvision
from torch.utils.data.dataloader import default_collate

from . import FairseqDataset


def _flatten(dico, prefix=None):
    """Flatten a nested dictionary."""
    new_dico = OrderedDict()
    if isinstance(dico, dict):
        prefix = prefix + '.' if prefix is not None else ''
        for k, v in dico.items():
            if v is None:
                continue
            new_dico.update(_flatten(v, prefix + k))
    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            new_dico.update(_flatten(v, prefix + '.[' + str(i) + ']'))
    else:
        new_dico = OrderedDict({prefix: dico})
    return new_dico


def _unflatten(dico):
    """Unflatten a flattened dictionary into a nested dictionary."""
    new_dico = OrderedDict()
    for full_k, v in dico.items():
        full_k = full_k.split('.')
        node = new_dico
        for k in full_k[:-1]:
            if k.startswith('[') and k.endswith(']'):
                k = int(k[1:-1])
            if k not in node:
                node[k] = OrderedDict()
            node = node[k]
        node[full_k[-1]] = v
    return new_dico


class NestedDictionaryDataset(FairseqDataset):

    def __init__(self, defn, sizes=None, precentage=1.0):
        super().__init__()
        self.defn = _flatten(defn)

        self.sizes = [sizes] if not isinstance(sizes, (list, tuple)) else sizes

        first = None
        for v in self.defn.values():
            if not isinstance(v, (FairseqDataset, torch.utils.data.Dataset,)):
                raise ValueError('Expected Dataset but found: {}'.format(v.__class__))
            first = first or v
            if len(v) > 0:
                assert len(v) == len(first), 'dataset lengths must match'
        self.inner_len = int(len(first) * precentage)
        self._len = len(first)

    def __getitem__(self, index):
        if index >= self.inner_len:
            index = index % self.inner_len
        return OrderedDict((k, ds[index]) for k, ds in self.defn.items())

    def __len__(self):
        return self.inner_len

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            return {}
        sample = OrderedDict()
        for k, ds in self.defn.items():
            try:
                sample[k] = ds.collater([s[k] for s in samples])
            except NotImplementedError:
                sample[k] = default_collate([s[k] for s in samples])
        return _unflatten(sample)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(s[index] for s in self.sizes)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if len(self.sizes) == 1:
            return self.sizes[0][index]
        else:
            return (s[index] for s in self.sizes)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return any(ds.supports_prefetch for ds in self.defn.values())

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        for ds in self.defn.values():
            if getattr(ds, 'supports_prefetch', False):
                ds.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.defn.values():
            ds.set_epoch(epoch)


class CIFAR10ColoredNestedDictionaryDataset(FairseqDataset):
    def __init__(self, train=True):
        super().__init__()
        # Read CIFAR10 using pytorch, download = True
        # The transforms should only include standard normalization to CIFAR10
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.507, 0.4865, 0.4409),
                                                                                      (0.2673, 0.2564, 0.2762))])
        self.dataset = torchvision.datasets.CIFAR10(root='./datalra/cifar10', train=True, download=True,
                                                    transform=transforms)

    def __getitem__(self, index):
        return {'id': index, 'net_input.src_tokens': self.dataset[index][0], 'net_input.src_lengths': 1024,
                'nsentences': 1, 'ntokens': 1024, 'target': self.dataset[index][1]}

    def __len__(self):
        return len(self.dataset)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return 1024 * 3

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            return {}
        sample = OrderedDict()
        sample['id'] = torch.tensor([s['id'] for s in samples])
        sample['nsentences'] = sum([s['nsentences'] for s in samples])
        sample['ntokens'] = sum([s['ntokens'] for s in samples])
        sample['net_input.src_lengths'] = torch.tensor([1024] * len(samples))
        sample['net_input.src_tokens'] = rearrange(torch.stack([s['net_input.src_tokens'] for s in samples]),
                                                   'b h l v -> b (h l v)')
        sample['target'] = torch.Tensor([s['target'] for s in samples]).long()
        return _unflatten(sample)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return 1024

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
        return any(ds.supports_prefetch for ds in self.defn.values())

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        # raise Exception('Oopsi')
        # for ds in self.defn.values():
        #     if getattr(ds, 'supports_prefetch', False):
        #         ds.prefetch(indices)
        pass

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        # for ds in self.defn.values():
        #     ds.set_epoch(epoch)
