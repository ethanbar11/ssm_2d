# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import Parameter


class RealNumberEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.Tensor(embedding_dim))
        self.bias = Parameter(torch.Tensor(embedding_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.weight + self.bias
        return emb

class CIFARColoredEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(3, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)

    def forward(self, x):
        x = x.view(x.size(0), -1, 3)
        emb = self.linear(x)
        return emb