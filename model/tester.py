import torch
import numpy as np
from collections import Counter
import os
from skimage.filters import threshold_otsu
from model.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from model.utils import classify_and_get_probs, compute_sam, plot_histogram
import rasterio
from tqdm import tqdm

class ModelTester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

    def load_model(self):
        """Load the pre-trained model."""
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.config.model["pretrained_name"])
        # model.load_state_dict(torch.load(os.path.join(self.config.paths["results_dir"], self.config.training["experiment_name"])))
        return model.to(self.device)

    def load_data(self, npz_file):
        """Load the preprocessed data for 2019 and 2020."""
        data = np.load(npz_file)
        return data["patches"], data["positions"], data["image_ids"]

    def run_damage_detection(self):
        """Load data, classify patches, and analyze changes."""
        # Load data
        patches_2019, positions_2019, image_ids_2019 = self.load_data(os.path.join(self.config.paths["preprocessed_test_dir_2019"], self.config.filenames["preprocessed"]))
        patches_2020, positions_2020, image_ids_2020 = self.load_data(os.path.join(self.config.paths["preprocessed_test_dir_2020"], self.config.filenames["preprocessed"]))

        if patches_2019.shape != patches_2020.shape:
            raise ValueError(f"Shape mismatch: patches from 2019 have shape {patches_2019.shape}, while patches from 2020 have shape {patches_2020.shape}")

        # Classify patches and get probabilities
        preds_2019, probs_2019 = classify_and_get_probs(patches_2019, self.model)
        preds_2020, probs_2020 = classify_and_get_probs(patches_2020, self.model)

        # Plot class distributions of classes for each year
        self.plot_class_distributions(preds_2019, preds_2020)

        # Analyze changes between the two years using different methods
        labels = self.analyze_changes(probs_2019, probs_2020)

        # Reconstruct images from patches and labels
        self.reconstruct_images(labels, positions_2020, image_ids_2020)

    def reconstruct_images(self, labels, positions, image_ids):
        """Reconstruct images from patches and labels and save them."""

        # Create dir if it doesn't exist
        path = os.path.join(self.config.paths['results_dir'], self.config.paths['results_damage_detection_dir'])
        os.makedirs(path, exist_ok=True)
        
        # Load channels order
        current_channels = self.config.channels["current"]
        selected_channels = self.config.channels["selected"]
        channels_order = [current_channels.index(c) + 1 for c in selected_channels]

        # Loop through image_ids with progress bar
        for image_id in tqdm(np.unique(image_ids), desc="Reconstructing images"):
            # Load the original image to get the shape
            path = os.path.join(self.config.paths['sentinel_data_dir'], self.config.filenames['images'].format(id=image_id))
            with rasterio.open(path) as src_image:
                image = src_image.read(channels_order)
                _, height, width = image.shape

                # Create an empty image
                result = np.zeros((height, width), dtype=np.uint8)
                
                # Take the labels and positions for image_id
                image_labels = labels[image_ids == image_id]
                image_positions = positions[image_ids == image_id]
                
                # Set 1 where label == 1
                for label, (row, col) in zip(image_labels, image_positions):
                    result[row, col] = label

                # Save the image
                output_path = os.path.join(path, self.config.filenames['damage_detection_image'].format(id=image_id))
                profile = src_image.profile
                profile.update({
                    'count': 1,
                    'dtype': result.dtype,
                    'nodata': 0
                })
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(result, 1)


    def plot_class_distributions(self, preds_2019, preds_2020):
        """Plot the distribution of classes for each year."""
        class_names = [
            "Agro-forestry areas",
            "Arable land",
            "Beaches, dunes, sands",
            "Broad-leaved forest",
            "Coastal wetlands",
            "Complex cultivation patterns",
            "Coniferous forest",
            "Industrial or commercial units",
            "Inland waters",
            "Inland wetlands",
            "Land principally occupied\nby agriculture, with significant\nareas of natural vegetation",
            "Marine waters",
            "Mixed forest",
            "Moors, heathland and\nsclerophyllous vegetation",
            "Natural grassland and\nsparsely vegetated areas",
            "Pastures",
            "Permanent crops",
            "Transitional woodland, shrub",
            "Urban fabric"
        ]
        # FIXME are them correct? https://bigearth.net/
        for year, preds in zip([2019, 2020], [preds_2019, preds_2020]):
            counts = Counter(preds)
            data_hist = [counts.get(i, 0) for i in range(19)]
            plot_histogram(
                data=data_hist,
                title=f"Distribution of classes ({year})",
                xlabel="Class",
                ylabel="Number of patches",
                xticks_labels=class_names,
                save_path=os.path.join(self.config.paths["results_dir"], self.config.filenames["class_distribution"].format(year=year))
            )

    def get_otsu_labels(self, probs_2019, probs_2020):
        """Compute Otsu's threshold for the given data."""
        if self.config.testing['distance_metric'] == "euclidean":
            scores = np.linalg.norm(probs_2019 - probs_2020, axis=1)
        elif self.config.testing['distance_metric'] == "sam":
            scores = compute_sam(probs_2019, probs_2020)
        else:
            raise ValueError(f"Unknown distance metric: {self.config.testing['distance_metric']}")
        otsu_threshold = threshold_otsu(scores)
        labels = (scores > otsu_threshold).astype(np.uint8)
        return labels


    def analyze_changes(self, probs_2019, probs_2020):
        """Analyze changes between two years."""
        labels = self.get_otsu_labels(probs_2019, probs_2020)
        plot_histogram(
            data=[np.sum(labels == 0), np.sum(labels == 1)],
            title=f"Healthy vs Damaged ({self.config.testing['distance_metric']} + Otsu)",
            xlabel="",
            ylabel="Number of patches",
            xticks_labels=["Healthy", "Damaged"],
            save_path=os.path.join(self.config.paths["results_dir"], "otsu_damaged.png")
        )
        return labels

