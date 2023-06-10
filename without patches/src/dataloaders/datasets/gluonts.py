try:
    from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
    from gluonts.dataset.util import to_pandas
except ImportError:
    _gluon_ts_missing = True

from torch.utils.data import Dataset

class GluonDataset(Dataset):

    def __init__(self, dataset_name, data_dir, **kwargs):
        self.dataset = get_dataset(dataset_name, path=data_dir, regenerate=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

