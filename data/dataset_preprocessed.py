import numpy as np
import torch
from preprocess import resize_image
from torch.utils.data import Dataset
import os

class NPZSentinelDataset(Dataset):
    """Dataset class for loading preprocessed Sentinel-2 patches from NPZ files."""
    def __init__(self, data_dir, target_size, resize_mode):
        """Initializes the dataset."""
        self.data_dir = data_dir
        self.target_size = target_size
        self.resize_mode = resize_mode

        # Load all blocks into memory
        self.block_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")
        ])
        self.loaded_blocks = {}
        for block_file in self.block_files:
            data = np.load(block_file)
            self.loaded_blocks[block_file] = {
                "patches": data["patches"],
                "labels": data["labels"]
            }

        # Build index mapping
        self.indices = []
        for i, block_file in enumerate(self.block_files):
            block_size = self.loaded_blocks[block_file]["patches"].shape[0]
            for j in range(block_size):
                self.indices.append((i, j))

    def __len__(self):
        """Returns the number of patches in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns a patch and label from the dataset."""
        # Get block file, block index and local index
        block_idx, local_idx = self.indices[idx]
        block_file = self.block_files[block_idx]

        # Load patch and label
        patch = self.loaded_blocks[block_file]["patches"][local_idx]
        patch = torch.from_numpy(patch).float()
        patch = resize_image(patch, (224,224), "pad") # FIXME config
        label = self.loaded_blocks[block_file]["labels"][local_idx]

        # Convert to PyTorch tensors
        label = torch.tensor(label).float().unsqueeze(0)
        
        return patch, label

    def clear_memory(self):
        """Clears loaded blocks from memory."""
        self.loaded_blocks.clear()
