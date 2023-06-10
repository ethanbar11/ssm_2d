#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Legacy entry point. Use fairseq_cli/train.py or fairseq-train instead.
"""
import torch

from fairseq_cli.train import cli_main


if __name__ == '__main__':
    # Set cuda visible in sys vars to 1 gpu
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cli_main()
