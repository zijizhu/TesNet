from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


class DogsDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None) -> None:
        super().__init__()
        assert split in ["train", "test"]
        self.split = split
        self.root = Path(root)
        train_mat = loadmat((self.root / "train_list.mat").as_posix(), squeeze_me=True)
        test_mat = loadmat((self.root / "test_list.mat").as_posix(), squeeze_me=True)
        self.samples = dict(
            train=train_mat['file_list'],
            test=test_mat['file_list']
        )
        self.lables = dict(
            train=train_mat['labels'] - 1,
            test=test_mat['labels'] - 1
        )
        self.transform = transform
        self.classes = np.unique(self.lables[self.split])

    def __len__(self):
        return len(self.samples[self.split])
    
    def __getitem__(self, index: int):
        im_path = self.samples[self.split][index]
        label = self.lables[self.split][index]
        im = Image.open(self.root / "Images" / im_path).convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, int(label)
