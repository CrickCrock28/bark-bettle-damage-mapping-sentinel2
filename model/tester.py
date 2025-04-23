import os
import torch
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from skimage.filters import threshold_otsu
from model.utils import classify_and_get_probs, compute_sam, compute_image_metrics
from model.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier


class ModelTester:
    def __init__(self, config, test_loaders):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_loaders = test_loaders
        self.model = self.load_model().to(self.device)

    def load_model(self):
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.config.model["pretrained_name"])
        # model.load_state_dict(torch.load(os.path.join(self.config.paths["results_dir"], self.config.training["experiment_name"])+".pth"))
        return model.to(self.device)

    def run_damage_detection(self):
        writer_path = os.path.join(self.config.paths["results_dir"], "damage_detection_results.xlsx")
        
        with pd.ExcelWriter(writer_path, engine="openpyxl") as writer:
            for mode in ["filtered", "all_data"]:
                loader_2019 = self.test_loaders[f"2019_{mode}_test"]
                loader_2020 = self.test_loaders[f"2020_{mode}_test"]
                metrics = self.evaluate_pair(loader_2019, loader_2020, mode)
                df = pd.DataFrame(metrics)
                df.to_excel(writer, sheet_name=mode, index=False)

    def evaluate_pair(self, loader_2019, loader_2020, mode):
        self.model.eval()
        results = []
        all_positions, all_image_ids, all_labels = [], [], []
        probs_2019_list, probs_2020_list = [], []

        for (patches_2019, labels_2019, positions_2019, image_ids_2019), (patches_2020, _, _, _) in tqdm(zip(loader_2019, loader_2020), total=len(loader_2019), desc=f"Evaluating {mode} pairs"):

            _, probs_2019 = classify_and_get_probs(patches_2019.numpy(), self.model)
            _, probs_2020 = classify_and_get_probs(patches_2020.numpy(), self.model)
            
            probs_2019_list.append(probs_2019)
            probs_2020_list.append(probs_2020)
            all_positions.extend(positions_2019)
            all_image_ids.extend(image_ids_2019)
            all_labels.extend(labels_2019.numpy())

        probs_2019_np = np.concatenate(probs_2019_list)
        probs_2020_np = np.concatenate(probs_2020_list)
        all_positions_np = np.array(all_positions)
        all_image_ids_np = np.array(all_image_ids)
        all_labels = np.array(all_labels)

        predicted_labels = self.get_otsu_labels(probs_2019_np, probs_2020_np)

        for image_id in tqdm(np.unique(all_image_ids_np), desc=f"Saving {mode} results"):

            mask = all_image_ids_np == image_id
            labels = predicted_labels[mask]
            ground_truth_labels = all_labels[mask]
            pos = all_positions_np[mask]

            pred_img = self.reconstruct_image(image_id, pos, labels)
            self.save_prediction_image(image_id, pred_img, mode)

            metrics = compute_image_metrics(ground_truth_labels, labels, image_id)
            results.append(metrics)

        return results

    def get_otsu_labels(self, probs_2019, probs_2020):
        if self.config.testing["distance_metric"] == "euclidean":
            scores = np.linalg.norm(probs_2019 - probs_2020, axis=1)
        elif self.config.testing["distance_metric"] == "sam":
            scores = compute_sam(probs_2019, probs_2020)
        else:
            raise ValueError("Unknown distance metric")
        threshold = threshold_otsu(scores)
        return (scores > threshold).astype(np.uint8)

    def reconstruct_image(self, img_id, positions, values):
        ref_path = os.path.join(self.config.paths["sentinel_data_dir_2020"], self.config.filenames["images"].format(id=img_id))
        with rasterio.open(ref_path) as src:
            _, h, w = src.read().shape
        image = np.zeros((h, w), dtype=np.uint8)
        for (r, c), v in zip(positions, values):
            image[r, c] = v
        return image

    def save_prediction_image(self, img_id, array, mode):
        output_dir = os.path.join(self.config.paths["results_dir"], f"damage_pred_{mode}")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, self.config.filenames["damage_detection_image"].format(id=img_id))

        ref_path = os.path.join(self.config.paths["sentinel_data_dir_2020"], self.config.filenames["images"].format(id=img_id))
        with rasterio.open(ref_path) as src:
            profile = src.profile
        profile.update({"count": 1, "dtype": array.dtype, "nodata": 0})
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(array, 1)