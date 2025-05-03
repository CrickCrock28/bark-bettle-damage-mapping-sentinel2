import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.utils import compute_image_metrics, reconstruct_image, save_prediction_image
from model.model import Pretrainedmodel
from model.utils import insert_images_into_excel

class ModelTester:
    """Class for testing the model on Sentinel data."""
    def __init__(self, config, test_loaders):
        """Initialize the tester with the given configuration and test loaders."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_loaders = test_loaders
        self.model = self.load_model().to(self.device)

    def load_model(self):
        """Load the saved model."""
        model = Pretrainedmodel.from_pretrained(
            self.config.model["pretrained_name"],
            num_classes = self.config.model["num_classes"]
        )
        weight_path = os.path.join(
            self.config.paths["results_dir"],
            self.config.paths["results_models_dir"],
            self.config.training["experiment_name"] + ".pth"
        )
        state_dict = torch.load(weight_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model.to(self.device)
    
    def run_test(self):
        """Run the test on the model."""
        writer_path = os.path.join(
            self.config.paths["results_dir"],
            self.config.filenames["metrics_results"]
        )
        test_result_base_path = os.path.join(
            self.config.paths["results_dir"],
            self.config.paths["results_test_dir"],
            self.config.training["experiment_name"]
        )
        results = []
        for mode in ["filtered", "all_data"]:
            loader = self.test_loaders[f"2020_{mode}_test"]
            metrics = self.evaluate(loader, mode)
            df = pd.DataFrame(metrics)
            results.append((mode, df))
        
        file_exists = os.path.exists(writer_path)
        with pd.ExcelWriter(writer_path, mode='a' if file_exists else 'w', engine='openpyxl', if_sheet_exists='replace' if file_exists else None) as writer:
            for mode, df in results:
                df.to_excel(writer, sheet_name=f"test_{mode}", index=False)

        insert_images_into_excel(writer_path, results, test_result_base_path, self.config.filenames["test_images"])

    def classify(self, patches):
        """Classify patches and return predictions."""

        patches = patches.to(self.device)

        # Classify the patches
        self.model.eval()
        with torch.no_grad():
            probs = self.model(patches)
            preds = torch.argmax(probs, dim=1)

        return preds.cpu().numpy()

    def evaluate(self, loader, mode):
        """Evaluate the model on a dataset."""
        results, all_positions, all_image_ids, all_labels, all_preds = [], [], [], [], []

        # Iterate through the data loader
        for (patches, labels, positions, image_ids) in tqdm(loader, total=len(loader), desc=f"Evaluating {mode} dataset"):

            # Use the model to classify the patches
            preds = self.classify(patches)
            
            # Save batch results
            all_positions.extend(positions)
            all_image_ids.extend(image_ids)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)

        # Concatenate the results from all batches
        all_positions_np = np.array(all_positions)
        all_image_ids_np = np.array(all_image_ids)
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        output_dir = os.path.join(
            self.config.paths["results_dir"],
            self.config.paths["results_test_dir"],
            self.config.training["experiment_name"],
            mode
        )
        # Iterate through unique image IDs
        for image_id in tqdm(np.unique(all_image_ids_np), desc=f"Saving results"):
            # Filter the results for the current image ID
            mask = all_image_ids_np == image_id
            labels = all_preds_np[mask]
            ground_truth_labels = all_labels_np[mask]
            pos = all_positions_np[mask]

            # Reconstruct the image and save it
            pred_img = reconstruct_image(self.config, image_id, pos, labels)
            save_prediction_image(self.config, output_dir, self.config.filenames["test_images"], image_id, pred_img)

            # Compute metrics for the image
            metrics = compute_image_metrics(ground_truth_labels, labels, image_id)
            results.append(metrics)

        return results
    