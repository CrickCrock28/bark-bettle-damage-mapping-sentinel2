import torch
import numpy as np
from collections import Counter
import os
from skimage.filters import threshold_otsu
from model.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from config.config_loader import Config
from model.utils import classify_and_get_probs, compute_sam, plot_histogram

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

    def load_data(self):
        """Load the preprocessed data for 2019 and 2020."""
        data_2019 = np.load(os.path.join(self.config.paths["preprocessed_test_dir_2019"], self.config.filenames["preprocessed"]))
        data_2020 = np.load(os.path.join(self.config.paths["preprocessed_test_dir_2020"], self.config.filenames["preprocessed"]))
        return data_2019["patches"], data_2020["patches"]

    def run_damage_detection(self):
        """Load data, classify patches, and analyze changes."""
        # Load data
        patches_2019, patches_2020 = self.load_data()

        if patches_2019.shape != patches_2020.shape:
            raise ValueError(f"Shape mismatch: patches from 2019 have shape {patches_2019.shape}, while patches from 2020 have shape {patches_2020.shape}")

        # Classify patches and get probabilities
        preds_2019, probs_2019 = classify_and_get_probs(patches_2019, self.model)
        preds_2020, probs_2020 = classify_and_get_probs(patches_2020, self.model)

        # Plot class distributions of classes for each year
        self.plot_class_distributions(preds_2019, preds_2020)

        # Analyze changes between the two years using different methods
        self.analyze_changes(probs_2019, probs_2020)

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

    def analyze_changes(self, probs_2019, probs_2020):
        """Analyze changes between two years using different methods."""
        # Use Otsu's method to find the optimal threshold for the Euclidean distance between the two years
        euclidean_scores = np.linalg.norm(probs_2020 - probs_2019, axis=1)
        otsu_euclidean = threshold_otsu(euclidean_scores)
        labels_euclidean = (euclidean_scores > otsu_euclidean).astype(np.uint8)
        plot_histogram(
            data=[np.sum(labels_euclidean == 0), np.sum(labels_euclidean == 1)],
            title="Healthy vs Damaged (Euclidean + Otsu)",
            xlabel="",
            ylabel="Number of patches",
            xticks_labels=["Healthy", "Damaged"],
            save_path=os.path.join(self.config.paths["results_dir"], "euclidean_otsu_damaged.png")
        )
        # Use Otsu's method to find the optimal threshold for the SAM distance between the two years
        sam_scores = compute_sam(probs_2019, probs_2020)
        otsu_sam = threshold_otsu(sam_scores)
        labels_sam = (sam_scores > otsu_sam).astype(np.uint8)
        plot_histogram(
            data=[np.sum(labels_sam == 0), np.sum(labels_sam == 1)],
            title="Healthy vs Damaged (SAM + Otsu)",
            xlabel="",
            ylabel="Number of patches",
            xticks_labels=["Healthy", "Damaged"],
            save_path=os.path.join(self.config.paths["results_dir"], "sam_otsu_damaged.png")
        )
