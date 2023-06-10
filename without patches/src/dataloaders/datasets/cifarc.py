from torch.utils.data import Dataset
from PIL import Image
from typing import Any, Callable, Optional, Tuple

class _CIFAR10C(Dataset):
    def __init__(self, data, targets, transform=None):
        assert data.shape[0] == targets.shape[0]
        self.data = data
        self.targets = targets
        self.transform = transform

    # copied from torchvision.datasets.cifar
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.data.shape[0]