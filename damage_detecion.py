import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from src.pretrain.reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
import warnings
from config.config_loader import Config
from tqdm import tqdm
import os

config = Config()

# FIXME workaround for the warning message
warnings.filterwarnings("ignore", message="Keyword 'img_size' unknown*")

# Labels for the classes
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

def classify_patches(patches, model):
    """Classify all patches at once using the pre-trained model, with progress bar."""
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Classify the patches
    model.eval()
    with torch.no_grad():
        batch_tensor = torch.from_numpy(patches).float().to(device)
        for _ in tqdm(range(1), desc="Classifying"):
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

    return preds.cpu().numpy()


def plot_histogram(data, title, xlabel, ylabel, xticks_labels=None, save_path=None):
    plt.figure(figsize=(12, 8) if xticks_labels else (5, 4))
    plt.bar(range(len(data)), data, tick_label=xticks_labels)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Load the preprocessed data
data_2019 = np.load(os.path.join(config.paths["preprocessed_test_dir_2019"], config.filenames["preprocessed"]))
data_2020 = np.load(os.path.join(config.paths["preprocessed_test_dir_2020"], config.filenames["preprocessed"]))

patches_2019 = data_2019["patches"]
patches_2020 = data_2020["patches"]

if patches_2019.shape != patches_2020.shape:
    raise ValueError(f"Shape mismatch: patches from 2019 have shape {patches_2019.shape}, while patches from 2020 have shape {patches_2020.shape}")

# Load the pre-trained model
model = BigEarthNetv2_0_ImageClassifier.from_pretrained(config.model["pretrained_name"])

# Classify the patches
classes_2019 = classify_patches(patches_2019, model)
classes_2020 = classify_patches(patches_2020, model)

# Get the changed patches counts
change_labels = (classes_2019 != classes_2020).astype(np.uint8)  # 1 = changed, 0 = unchanged
changed_counts = Counter(change_labels)

class_counts_2019 = Counter(classes_2019)
data_2019_hist = [class_counts_2019[i] for i in range(19)]
plot_histogram(
    data=data_2019_hist,
    title="Distribution of classes (2019)",
    xlabel="Class",
    ylabel="Number of patches",
    xticks_labels=[class_names[i] for i in range(19)],
    save_path=os.path.join(config.paths["results_dir"], config.filenames["class_distribution"].format(year=2019))
)

class_counts_2020 = Counter(classes_2020)
data_2020_hist = [class_counts_2020[i] for i in range(19)]
plot_histogram(
    data=data_2020_hist,
    title="Distribution of classes (2020)",
    xlabel="Class",
    ylabel="Number of patches",
    xticks_labels=[class_names[i] for i in range(19)],
    save_path=os.path.join(config.paths["results_dir"], config.filenames["class_distribution"].format(year=2020))
)

plot_histogram(
    data=[changed_counts[0], changed_counts[1]],
    title="Patch Damaged vs Healthy",
    xlabel="",
    ylabel="Number of patches",
    xticks_labels=["Healthy", "Damaged"],
    save_path=os.path.join(config.paths["results_dir"], config.filenames["damaged_vs_healthy"])
)

