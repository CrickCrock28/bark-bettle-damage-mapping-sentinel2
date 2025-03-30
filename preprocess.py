import os
import torch
import numpy as np
import rasterio
from tqdm import tqdm
import time
import torch.nn as nn

from config.config_loader import Config

def extract_pixel_patch(img_tensor, pixel_idx, radius):
    """Extract a patch centered at a pixel index from an image tensor."""

    # Get pixel coordinates
    channels, height, width = img_tensor.shape
    row = pixel_idx // width
    column = pixel_idx % width

    # Initialize patch and valid mask
    patch = torch.zeros(channels, 2 * radius + 1, 2 * radius + 1).float()
    valid_mask = torch.zeros(2 * radius + 1, 2 * radius + 1).bool()
    
    # Extract patch centered at pixel
    out_of_bounds = False
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            x = row + i
            y = column + j

            # If pixel is within bounds, add it to the patch and set the valid mask
            if 0 <= x < height and 0 <= y < width:
                patch[:, i + radius, j + radius] = img_tensor[:, x, y]
                valid_mask[i + radius, j + radius] = True
            else:
                out_of_bounds = True
    
    # Fill pixel out of bounds
    if out_of_bounds:
        # Compute mean values for each channel
        mean_values = torch.zeros(channels).float()
        for channel in range(channels):
            valid_vals = patch[channel][valid_mask]
            mean_values[channel] = valid_vals.mean()
        # Fill out of bounds pixels with mean values
        for channel in range(channels):
            patch[channel][~valid_mask] = mean_values[channel]

    return patch

def get_mask_ids(mask_dir, mask_filename_format):
    """Get the IDs of the masks in the mask directory."""
    return [
        f.split("_")[1].split(".")[0]
        for f in os.listdir(mask_dir)
        if f.startswith(mask_filename_format.split("{")[0]) 
        and f.endswith(mask_filename_format.split("}")[1])
    ]

def load_data(mask_dir, sentinel_data_dirs, mask_filename_format, image_filename_format):
    """Load the paths of the images and masks in the dataset."""

    # Get mask IDs
    mask_ids = get_mask_ids(mask_dir, mask_filename_format)
    
    # Find image paths for each mask
    image_paths, mask_paths = [], []
    for mask_id in mask_ids:
        for data_dir in sentinel_data_dirs:
            candidate = os.path.join(data_dir, image_filename_format.format(id=mask_id))
            if os.path.exists(candidate):
                image_paths.append(candidate)
                mask_paths.append(os.path.join(mask_dir, mask_filename_format.format(id=mask_id)))
                break

    return image_paths, mask_paths

def save_block(output_dir, patches, labels, block_idx, filename_format):
    """Save a block of patches to a .npz file."""
    filename = filename_format.format(id=block_idx)
    block_filepath = os.path.join(output_dir, filename)
    np.savez_compressed(block_filepath, patches=np.array(patches), labels=np.array(labels))

def pad_to_target_size(img_tensor, target_size):
    """Pads an image tensor to the target size."""

    _, height, width = img_tensor.shape
    padding_height = target_size[0] - height
    padding_width = target_size[1] - width

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

def resize_image(image_tensor, target_size, resize_mode):
    """Resize an image tensor to the target size."""

    if resize_mode == "interpolate":
        image_tensor = nn.Upsample(
            size=target_size,
            mode="bilinear",
            align_corners=False # FIXME check if this bool is correct
        )(image_tensor.unsqueeze(0)).squeeze(0)
    elif resize_mode == "pad":
        image_tensor = pad_to_target_size(image_tensor, target_size)
    else:
        raise ValueError("Invalid resize mode")
    
    return image_tensor

def preprocess_images(config, image_paths, mask_paths, channels_order, output_dir):
    """Preprocess images and masks, extracting patches and saving them to .npz files."""
    os.makedirs(output_dir, exist_ok=True)
    patch_buffer = []
    label_buffer = []
    block_idx = 0
    total_patches = 0

    # Iterate over images and masks
    for idx, (image_path, mask_path) in enumerate(tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Preprocessing images")):

        # Open image and mask
        with rasterio.open(image_path) as src_image, rasterio.open(mask_path) as src_mask:
            image = src_image.read(channels_order)
            mask = src_mask.read(1) 
            height, width = src_image.shape

            image_tensor = torch.from_numpy(image).float()/10000
            mask_tensor = torch.from_numpy(mask).float()

            # Iterate over pixels
            for row in range(height):
                for col in range(width):
                    # Extract patch and label
                    pixel_idx = row * width + col
                    patch = extract_pixel_patch(image_tensor, pixel_idx, config.dataset["radius"])
                    # patch_resized = resize_image(patch, config.dataset["target_size"], config.dataset["resize_mode"])
                    patch_buffer.append(patch.numpy())
                    label = mask_tensor[row, col].item()
                    label_buffer.append(label)
                    total_patches += 1

                    # Save block if buffer is full
                    if len(patch_buffer) >= config.dataset["block_size"]:
                        save_block(output_dir, patch_buffer, label_buffer, block_idx, config.filenames["block"])
                        patch_buffer.clear()
                        label_buffer.clear()
                        block_idx += 1
    
    # Save remaining patches
    if patch_buffer:
        save_block(output_dir, patch_buffer, label_buffer, block_idx, config.filenames["block"])

    print(f"\nCompleted preprocessing. Total patches: {total_patches}")

def main():
    """Main function for preprocessing the dataset."""
    config = Config()
    now = time.time()

    # Load configuration parameters
    current_channels = config.channels["current"]
    selected_channels = config.channels["selected"]
    channels_order = [current_channels.index(c) + 1 for c in selected_channels]

    output_train_dir = config.paths["preprocessed_train_dir"]
    output_test_dir = config.paths["preprocessed_test_dir"]

    # Load train dataset
    train_images, train_masks = load_data(
        config.paths["train_mask_dir"],
        config.paths["sentinel_data_dirs"],
        config.filenames["masks"],
        config.filenames["images"]
    )

    # Preprocess train dataset
    print("\nStarting preprocessing train dataset...")
    preprocess_images(config, train_images, train_masks, channels_order, output_train_dir)

    # Load test dataset
    test_images, test_masks = load_data(
        config.paths["test_mask_dir"],
        config.paths["sentinel_data_dirs"],
        config.filenames["masks"],
        config.filenames["images"]
    )

    # Preprocess test dataset   
    print("\nStarting preprocessing test dataset...")
    preprocess_images(config, test_images, test_masks, channels_order, output_test_dir)

    print("\nPreprocessing completed.")
    print(f"Elapsed time: {time.time() - now:.2f} seconds")

if __name__ == "__main__":
    main()
