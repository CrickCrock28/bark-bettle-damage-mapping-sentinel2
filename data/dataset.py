import numpy as np
import torch
from data.preprocess import resize_image
from torch.utils.data import Dataset
import os

class NPZSentinelDataset(Dataset):
    """Dataset class for loading preprocessed Sentinel-2 patches from a single NPZ file."""
    def __init__(self, data_dir, target_size, resize_mode, preprocessed_filename):
        """Initializes the dataset."""
        self.preprocessed_filename = preprocessed_filename
        self.target_size = target_size
        self.resize_mode = resize_mode

        # Load the single npz file
        file_path = os.path.join(data_dir, preprocessed_filename)
        data = np.load(file_path)
        self.patches = data["patches"]
        self.labels = data["labels"]

    def __len__(self):
        """Returns the number of patches in the dataset."""
        return len(self.patches)

    def __getitem__(self, idx):
        """Returns a patch and label from the dataset."""
        patch = torch.from_numpy(self.patches[idx]).float()
        patch = resize_image(patch, self.target_size, self.resize_mode)
        label = torch.tensor(int(self.labels[idx])).long()
        return patch, label

    def clear_memory(self):
        """Clears loaded data from memory."""
        del self.patches
        del self.labels
