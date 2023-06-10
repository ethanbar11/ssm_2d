import logging
import os

import numpy as np

from fairseq import utils
from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PixelSequenceDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task

logger = logging.getLogger(__name__)


@register_task('cifar10')
class Cifar10(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes or regression targets')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--pixel-normalization', type=float, nargs='+', default=None,
                            help='mean and std for pixel normalization.')

    def __init__(self, args):
        super().__init__(args)
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions
        self.normalization = (0.5, 0.5) if args.pixel_normalization is None else args.pixel_normalization
        assert len(self.normalization) == 2

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        raise NotImplementedError

    @classmethod
    def setup_task(cls, args, **kwargs):
        return Cifar10(args)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, percentage=1.0):
            split_path = get_path(type, split)
            dataset = PixelSequenceDataset(split_path + '.src', self.normalization)
            return dataset

        precentage = self.args.dataset_percentage if split == 'train' else 1.0
        src_ds = make_dataset('input')
        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(int(len(src_ds) * precentage))

        src_tokens = TruncateDataset(src_ds, self.args.max_positions)
        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }
        label_path = get_path('label', split) + '.label'
        if os.path.exists(label_path):
            label_dataset = RawLabelDataset([int(line.strip()) for i, line in enumerate(open(label_path).readlines())])
            dataset.update(target=label_dataset)

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
            precentage=precentage
        )
        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        return model

    def max_positions(self):
        return self._max_positions

    @property
    def target_dictionary(self):
        return None
