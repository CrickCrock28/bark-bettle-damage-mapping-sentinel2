import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import threshold_otsu
from model.utils import compute_sam, compute_sam_2, compute_image_metrics, reconstruct_image, save_prediction_image, insert_images_into_excel
from model.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

class DamageDetectionTester:
    """Class for testing the model on Sentinel data."""
    def __init__(self, config, test_loaders):
        """Initialize the tester with the given configuration and test loaders."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_loaders = test_loaders
        self.model = self.load_model().to(self.device)

    def load_model(self):
        """Load the model from the specified path and delete the last layer."""
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.config.model["pretrained_name"])
        model.model.vision_encoder.fc = torch.nn.Identity()
        return model.to(self.device)

    def run_damage_detection(self):
        """Run damage detection on the test datasets and save results."""
        writer_path = os.path.join(
            self.config.paths["results_dir"],
            self.config.filenames["metrics_results"]
        )
        damage_detection_result_base_path = os.path.join(
            self.config.paths["results_dir"],
            self.config.paths["results_damage_detection_dir"]
        )
        
        for mode in ["filtered", "all_data"]:
            sheet_name = f"dam-det_{mode}_{self.config.testing['distance_metric']}"
            loader_2019 = self.test_loaders[f"2019_{mode}_test"]
            loader_2020 = self.test_loaders[f"2020_{mode}_test"]
            metrics = self.detect_damage(loader_2019, loader_2020, mode)
            df = pd.DataFrame(metrics)
            
            file_exists = os.path.exists(writer_path)
            with pd.ExcelWriter(writer_path, mode='a' if file_exists else 'w', engine='openpyxl', if_sheet_exists='replace' if file_exists else None) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            insert_images_into_excel(
                writer_path,
                df,
                sheet_name,
                os.path.join(damage_detection_result_base_path, mode),
                self.config.filenames["damage_detection_images"],
                self.config.paths["test_mask_dir"],
                self.config.filenames["masks"]
            )

    def detect_damage(self, loader_2019, loader_2020, mode):
        """Evaluate the model on a pair of datasets (2019 and 2020)."""
        results, all_positions, all_image_ids, all_labels, probs_2019_list, probs_2020_list = [], [], [], [], [], []

        # Iterate through the 2019 and 2020 loaders simultaneously
        for (patches_2019, labels_2019, positions_2019, image_ids_2019), (patches_2020, _, _, image_ids_2020) in tqdm(zip(loader_2019, loader_2020), total=len(loader_2019), desc=f"Evaluating {mode} pairs"):

            # Ensure that id is the same for both loaders
            if image_ids_2019 != image_ids_2020:
                raise ValueError("Image IDs do not match between 2019 and 2020 loaders.")

            # Use the model to classify the patches and get probabilities
            probs_2019 = self.classify(patches_2019)
            probs_2020 = self.classify(patches_2020)
            
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

        output_dir = os.path.join(
            self.config.paths["results_dir"],
            self.config.paths["results_damage_detection_dir"],
            mode
        )
        # Iterate through unique image IDs
        for image_id in tqdm(np.unique(all_image_ids_np), desc=f"Saving {mode} results"):
            # Filter the results for the current image ID
            mask = all_image_ids_np == image_id
            labels = predicted_labels[mask]
            ground_truth_labels = all_labels[mask]
            pos = all_positions_np[mask]

            # Reconstruct the image and save it
            pred_img = reconstruct_image(self.config, image_id, pos, labels)
            total_pixels = np.prod(pred_img.shape[:])
            save_prediction_image(self.config, output_dir, self.config.filenames["damage_detection_images"], image_id, pred_img)

            # Compute metrics for the image
            if len(labels) < total_pixels:
                labels = np.concatenate([labels, np.zeros(total_pixels - len(labels))])
                ground_truth_labels = np.concatenate([ground_truth_labels, np.zeros(total_pixels - len(ground_truth_labels))])
            metrics = compute_image_metrics(ground_truth_labels, labels, image_id)
            results.append(metrics)

        return results

    def classify(self, patches):
        """Classify the patches using the model."""
        self.model.eval()
        with torch.no_grad():
            batch_tensor = patches.float().to(self.device)
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def get_otsu_labels(self, probs_2019, probs_2020):
        """"Get the predicted labels using Otsu's method."""
        if self.config.testing["distance_metric"] == "euclidean":
            scores = np.linalg.norm(probs_2019 - probs_2020, axis=1)
        elif self.config.testing["distance_metric"] == "sam":
            scores = compute_sam(probs_2019, probs_2020)
            # FIXME choose one of the two functions and delete comments
            # scores2 = compute_sam_2(probs_2019, probs_2020)
            # print("Massima differenza", np.max(np.abs(scores - scores2))) # 3.189500421285629e-05
            # print("Differenza media", np.mean(np.abs(scores - scores2))) # 1.2524034241127164e-06
        else:
            raise ValueError("Unknown distance metric, check the config file.")
        threshold = threshold_otsu(scores)
        return (scores > threshold).astype(np.uint8)
