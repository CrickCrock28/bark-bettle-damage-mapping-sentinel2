import os
import torch
import numpy as np
import rasterio
from tqdm import tqdm
import torch.nn as nn

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

def should_keep_patch(row, col, mask_tensor, forest_tensor):
    """Return True if the pixel is forest or if the label is 1."""
    label = mask_tensor[row, col].item()
    is_forest = forest_tensor[row, col].item()
    return is_forest != 0 or label == 1

def get_mask_ids(mask_dir, mask_filename_format):
    """Get the IDs of the masks in the mask directory."""
    return [
        f.split("_")[1].split(".")[0]
        for f in os.listdir(mask_dir)
        if f.startswith(mask_filename_format.split("{")[0]) 
        and f.endswith(mask_filename_format.split("}")[1])
    ]

def load_data(sentinel_data_dir, image_filename_format, mask_dir, mask_filename_format, forest_mask_dir=None, forest_mask_filename_format=None):
    """Load the paths of the images, masks, and optionally forest masks in the dataset."""

    # Get mask IDs
    mask_ids = get_mask_ids(mask_dir, mask_filename_format)

    # Find image paths for each mask
    image_paths, mask_paths, forest_paths = [], [], []

    for mask_id in mask_ids:
        image_path = os.path.join(sentinel_data_dir, image_filename_format.format(id=mask_id))
        mask_path = os.path.join(mask_dir, mask_filename_format.format(id=mask_id))

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found in {sentinel_data_dir}")

        image_paths.append(image_path)
        mask_paths.append(mask_path)

        if forest_mask_dir:
            forest_path = os.path.join(forest_mask_dir, forest_mask_filename_format.format(id=mask_id))
            if not os.path.exists(forest_path):
                raise FileNotFoundError(f"Forest mask file {forest_path} not found in {forest_mask_dir}")
            forest_paths.append(forest_path)

    return image_paths, mask_paths, forest_paths if forest_mask_dir else None

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
            align_corners=True
        )(image_tensor.unsqueeze(0)).squeeze(0)
    elif resize_mode == "pad":
        image_tensor = pad_to_target_size(image_tensor, target_size)
    elif resize_mode == "none":
        pass
    else:
        raise ValueError("Invalid resize mode")

    return image_tensor

def preprocess_images(config, channels_order, output_dir, image_paths, mask_paths, forest_paths=None):
    """Preprocess images and masks, extracting patches and saving them in a single .npz file."""

    os.makedirs(output_dir, exist_ok=True)
    patches = []
    labels = []

    use_forest = forest_paths is not None

    # Create a zip iterator for images, masks, and (if present) forest masks
    if forest_paths:
        iterator = zip(image_paths, mask_paths, forest_paths)
    else:
        iterator = zip(image_paths, mask_paths)

    # Iterate over images, masks, and (if present) forest masks
    for idx, items in enumerate(tqdm(iterator, total=len(image_paths), desc="Preprocessing images")):
        image_path = items[0]
        mask_path = items[1]

        # Open image and mask (and forest mask if present)
        with rasterio.open(image_path) as src_image, \
             rasterio.open(mask_path) as src_mask:

            image = src_image.read(channels_order)
            mask = src_mask.read(1) 
            height, width = src_image.shape

            image_tensor = torch.from_numpy(image).float() / 10000
            mask_tensor = torch.from_numpy(mask).float()
            if use_forest:
                forest_path = items[2]
                with rasterio.open(forest_path) as src_forest:
                    forest_mask = src_forest.read(1)
                    forest_tensor = torch.from_numpy(forest_mask).float()

            # Iterate over pixels
            for row in range(height):
                for col in range(width):
                    # # Alternate pixels
                    # if (row + col) % 2 != 0:
                    #     continue

                    # Apply filtering condition: keep only if forest or label == 1
                    if use_forest and not should_keep_patch(row, col, mask_tensor, forest_tensor):
                        continue

                    # Extract patch and label
                    pixel_idx = row * width + col
                    patch = extract_pixel_patch(image_tensor, pixel_idx, config.dataset["radius"])
                    label = mask_tensor[row, col].item()
                    patches.append(patch.numpy())
                    labels.append(label)

    # Save all patches and labels in a single .npz file
    block_filepath = os.path.join(output_dir, config.filenames["preprocessed"])
    np.savez_compressed(block_filepath, patches=np.array(patches), labels=np.array(labels))
