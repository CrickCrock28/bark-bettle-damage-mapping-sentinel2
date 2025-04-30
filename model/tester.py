import os
import torch
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from skimage.filters import threshold_otsu
from model.utils import classify_and_get_probs, compute_sam, compute_image_metrics
from model.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
import matplotlib.pyplot as plt

class ModelTester:
    """Class for testing the model on Sentinel data."""
    def __init__(self, config, test_loaders):
        """Initialize the tester with the given configuration and test loaders."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_loaders = test_loaders
        self.model = self.load_model().to(self.device)

    def load_model(self):
        """Load the model from the specified path."""
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.config.model["pretrained_name"])
        # model.load_state_dict(torch.load(os.path.join(self.config.paths["results_dir"], self.config.training["experiment_name"])+".pth"))
        return model.to(self.device)

    def run_damage_detection(self):
        """Run damage detection on the test datasets and save results."""
        writer_path = os.path.join(self.config.paths["results_dir"], "damage_detection_results.xlsx")
        
        results = []
        for mode in ["filtered", "all_data"]:
            loader_2019 = self.test_loaders[f"2019_{mode}_test"]
            loader_2020 = self.test_loaders[f"2020_{mode}_test"]
            metrics = self.evaluate_pair(loader_2019, loader_2020, mode)
            df = pd.DataFrame(metrics)
            results.append((mode, df))
            
        with pd.ExcelWriter(writer_path, engine="openpyxl") as writer:
            for mode, df in results:
                df.to_excel(writer, sheet_name=mode, index=False)

    def evaluate_pair(self, loader_2019, loader_2020, mode):
        """Evaluate the model on a pair of datasets (2019 and 2020)."""
        results = []
        all_positions, all_image_ids, all_labels = [], [], []
        probs_2019_list, probs_2020_list = [], []

        # Iterate through the 2019 and 2020 loaders simultaneously
        for (patches_2019, labels_2019, positions_2019, image_ids_2019), (patches_2020, _, _, image_ids_2020) in tqdm(zip(loader_2019, loader_2020), total=len(loader_2019), desc=f"Evaluating {mode} pairs"):

            # Be sure that id is the same for both loaders
            if image_ids_2019 != image_ids_2020:
                raise ValueError("Image IDs do not match between 2019 and 2020 loaders.")

            # Use the model to classify the patches and get probabilities
            _, probs_2019 = classify_and_get_probs(patches_2019.numpy(), self.model)
            _, probs_2020 = classify_and_get_probs(patches_2020.numpy(), self.model)
            
            # Save batch results
            probs_2019_list.append(probs_2019)
            probs_2020_list.append(probs_2020)
            all_positions.extend(positions_2019)
            all_image_ids.extend(image_ids_2019)
            all_labels.extend(labels_2019.numpy())

        # Concatenate the results from all batches
        probs_2019_np = np.concatenate(probs_2019_list)
        probs_2020_np = np.concatenate(probs_2020_list)
        all_positions_np = np.array(all_positions)
        all_image_ids_np = np.array(all_image_ids)
        all_labels = np.array(all_labels)

        # Get the predicted labels using Otsu's method
        predicted_labels = self.get_otsu_labels(probs_2019_np, probs_2020_np)

        # Iterate through unique image IDs
        for image_id in tqdm(np.unique(all_image_ids_np), desc=f"Saving {mode} results"):
            # Filter the results for the current image ID
            mask = all_image_ids_np == image_id
            labels = predicted_labels[mask]
            ground_truth_labels = all_labels[mask]
            pos = all_positions_np[mask]

            # Reconstruct the image and save it
            pred_img = self.reconstruct_image(image_id, pos, labels)
            self.save_prediction_image(image_id, pred_img, mode)

            # Compute metrics for the image
            metrics = compute_image_metrics(ground_truth_labels, labels, image_id)
            results.append(metrics)

        return results

    def get_otsu_labels(self, probs_2019, probs_2020):
        """"Get the predicted labels using Otsu's method."""
        if self.config.testing["distance_metric"] == "euclidean":
            scores = np.linalg.norm(probs_2019 - probs_2020, axis=1)
        elif self.config.testing["distance_metric"] == "sam":
            scores = compute_sam(probs_2019, probs_2020)
        else:
            raise ValueError("Unknown distance metric")
        threshold = threshold_otsu(scores)
        return (scores > threshold).astype(np.uint8)

    def reconstruct_image(self, img_id, positions, values):
        """Reconstruct the image from the predicted labels."""
        reference_path = os.path.join(
            self.config.paths["sentinel_data_dir_2020"],
            self.config.filenames["images"].format(id=img_id)
        )
        with rasterio.open(reference_path) as src:
            _, h, w = src.read().shape
        image = np.zeros((h, w), dtype=np.uint8)
        for (r, c), v in zip(positions, values):
            image[r, c] = v
        return image

    def save_prediction_image(self, img_id, array, mode):
        """Save the predicted image."""
        output_dir = os.path.join(
            self.config.paths["results_dir"],
            f"damage_pred_{mode}"
        )
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(
            output_dir,
            self.config.filenames["damage_detection_image"].format(id=img_id)
        )

        # Copy the profile from the reference image
        reference_path = os.path.join(
            self.config.paths["sentinel_data_dir_2020"],
            self.config.filenames["images"].format(id=img_id)
        )
        with rasterio.open(reference_path) as src:
            profile = src.profile

        # Update the profile for the output image and save it
        profile.update(
            {
                "count": 1,
                "dtype": array.dtype,
                "nodata": 0
            }
        )
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(array, 1)