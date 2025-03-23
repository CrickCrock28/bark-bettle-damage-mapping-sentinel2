import os
import torch
import rasterio
from torch.utils.data import Dataset
import torch.nn as nn
import random
from tqdm import tqdm

class SentinelDataset(Dataset):
    def __init__(self, image_paths, mask_paths, channels_order, target_size=(224, 224), radius=1, sampling_rate=1.0):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.channels_order = channels_order
        self.target_size = target_size
        self.radius = radius
        self.sampling_rate = sampling_rate

        self.pixel_indices = []

        for i, image_path in enumerate(self.image_paths):
            with rasterio.open(image_path) as src:
                h, w = src.shape
                indices = [(i, row, col) for row in range(h) for col in range(w)]
                self.pixel_indices.extend(indices)

        if sampling_rate < 1.0:
            num_samples = int(len(self.pixel_indices) * sampling_rate)
            self.pixel_indices = random.sample(self.pixel_indices, num_samples)

    def __len__(self):
        return len(self.pixel_indices)

    def __getitem__(self, idx):
        image_idx, row, col = self.pixel_indices[idx]

        image_path = self.image_paths[image_idx]
        mask_path = self.mask_paths[image_idx]

        with rasterio.open(image_path) as src:
            image = src.read(self.channels_order)
            image_tensor = torch.from_numpy(image).float() / 10000.0
            height, width = src.shape
            pixel_idx = row * width + col
            patch = extract_pixel_patch(image_tensor, pixel_idx, self.radius)
            patch = pad_to_target_size(patch, self.target_size)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            mask_tensor = torch.from_numpy(mask).float()
            label = mask_tensor[row, col]

        return patch, label.unsqueeze(0)
    
    def get_labels_counts(self):
        counts = {0: 0, 1: 0}

        mask_cache = {}

        for image_idx, row, col in tqdm(self.pixel_indices, desc="Counting labels"):
            if image_idx not in mask_cache:
                mask_path = self.mask_paths[image_idx]
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                mask_cache[image_idx] = mask
            
            mask = mask_cache[image_idx]
            label = mask[row, col]

            if label == 0:
                counts[0] += 1
            elif label == 1:
                counts[1] += 1
            else:
                print(f"Warning: Unexpected label {label} at image {image_idx}, row {row}, col {col}")

        return counts
  
def extract_pixel_patch(img_tensor, pixel_idx, radius):
    channels, height, width = img_tensor.shape
    row = pixel_idx // width
    column = pixel_idx % width

    patch = torch.zeros(channels, 2 * radius + 1, 2 * radius + 1).float()
    valid_mask = torch.zeros(2 * radius + 1, 2 * radius + 1).bool()
    
    border = False
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            x = row + i
            y = column + j

            if 0 <= x < height and 0 <= y < width:
                patch[:, i + radius, j + radius] = img_tensor[:, x, y]
                valid_mask[i + radius, j + radius] = True
            else:
                border = True
        
    if border:
        mean_values = torch.zeros(channels).float()
        for channel in range(channels):
            valid_vals = patch[channel][valid_mask]
            if valid_vals.numel() > 0:
                mean_values[channel] = valid_vals.mean()

        for channel in range(channels):
            patch[channel][~valid_mask] = mean_values[channel]

    return patch

def pad_to_target_size(img_tensor, target_size):
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
    return [
        f.split("_")[1].split(".")[0]
        for f in os.listdir(mask_dir)
        if f.startswith(mask_filename_format.split("{")[0]) 
        and f.endswith(mask_filename_format.split("}")[1])
    ]

def load_data(mask_dir, sentinel_data_dirs, mask_filename_format, image_filename_format):
    mask_ids = get_mask_ids(mask_dir, mask_filename_format)
    
    image_paths, mask_paths = [], []
    for mask_id in mask_ids:
        for data_dir in sentinel_data_dirs:
            candidate = os.path.join(data_dir, image_filename_format.format(id=mask_id))
            if os.path.exists(candidate):
                image_paths.append(candidate)
                mask_paths.append(os.path.join(mask_dir, mask_filename_format.format(id=mask_id)))
                break

    return image_paths, mask_paths
