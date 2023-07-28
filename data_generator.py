import numpy as np
from torch.utils.data import Dataset


class maestroDataset(Dataset):
    def __init__(self, data, target, category_target, place_target):
        """Initialize the dataset loading."""
        self.data = data
        self.target = target
        self.category_target = category_target
        self.place_target = place_target
        
    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i):
        #print(f'get item: {i}')
        data = self.data[i].astype(np.float32)
        target = self.target[i].astype(np.float32)
        category_target = self.category_target[i].astype(np.float32)
        place_target = self.place_target[i].astype(np.float32)

        return data, target, category_target, place_target

