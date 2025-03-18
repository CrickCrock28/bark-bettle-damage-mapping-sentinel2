import os
import torch
import rasterio
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn as nn

class SentinelDataset(Dataset):
    def __init__(self, image_paths, mask_paths, channels_order, target_size=(224, 224)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.channels_order = channels_order
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        with rasterio.open(image_path) as src:
            image = src.read(self.channels_order)
            image = torch.from_numpy(image).float()

        image = self.pad_to_target_size(image, self.target_size)
        image = TF.resize(image, self.target_size)

        with rasterio.open(mask_path) as src:
            mask = src.read()
            # To have a simple binary classification problem, we consider the mask as positive if it has at least one pixel with value 1
            label = torch.tensor([1.0 if mask.sum() > 0 else 0.0], dtype=torch.float32)

        return image, label

    def pad_to_target_size(self, img_tensor, target_size):
        _, h, w = img_tensor.shape
        padding_height = max(target_size[0] - h, 0)
        padding_width = max(target_size[1] - w, 0)

        if padding_height > 0 or padding_width > 0:
            img_tensor = nn.functional.pad(
                img_tensor,
                pad=(
                    padding_width // 2,
                    padding_width - padding_width // 2,
                    padding_height // 2,
                    padding_height - padding_height // 2
                ),
                mode='constant',
                value=0
            )
        return img_tensor

def get_mask_ids(mask_dir, mask_filename_format):
    return [f.split("_")[1].split(".")[0] for f in os.listdir(mask_dir) if f.startswith(mask_filename_format.split("{")[0]) and f.endswith(mask_filename_format.split("}")[1])]

def load_data(mask_dir, sentinel_data_dirs, mask_filename_format, image_filename_format):
    ids = get_mask_ids(mask_dir, mask_filename_format)
    image_paths, mask_paths = [], []

    for id in ids:
        for data_dir in sentinel_data_dirs:
            candidate = os.path.join(data_dir, image_filename_format.format(id=id))
            if os.path.exists(candidate):
                image_paths.append(candidate)
                mask_paths.append(os.path.join(mask_dir, mask_filename_format.format(id=id)))
                break

    return image_paths, mask_paths
