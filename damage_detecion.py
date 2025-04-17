import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from src.pretrain.reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
import warnings
from config.config_loader import Config
import os
from skimage.filters import threshold_otsu

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

def classify_and_get_probs(patches, model):
    """Classify patches and return both predictions and probabilities."""

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Classify the patches and get probabilities
    model.eval()
    with torch.no_grad():
        batch_tensor = torch.from_numpy(patches).float().to(device)
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.cpu().numpy(), probs.cpu().numpy()

def plot_histogram(data, title, xlabel, ylabel, xticks_labels=None, save_path=None):
    """Plot a histogram of the data."""
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(data)), data, tick_label=xticks_labels)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + max(data)*0.01, f'{int(yval)}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_sam(v1, v2):
    """Compute the Spectral Angle Mapper (SAM) between two sets of vectors."""
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    cos_angle = np.clip(dot / (norm1 * norm2 + 1e-8), -1.0, 1.0)
    return np.arccos(cos_angle)

# Load the preprocessed data
data_2019 = np.load(os.path.join(config.paths["preprocessed_test_dir_2019"], config.filenames["preprocessed"]))
data_2020 = np.load(os.path.join(config.paths["preprocessed_test_dir_2020"], config.filenames["preprocessed"]))

patches_2019 = data_2019["patches"]
patches_2020 = data_2020["patches"]

if patches_2019.shape != patches_2020.shape:
    raise ValueError(f"Shape mismatch: patches from 2019 have shape {patches_2019.shape}, while patches from 2020 have shape {patches_2020.shape}")

# Load the pre-trained model
model = BigEarthNetv2_0_ImageClassifier.from_pretrained(config.model["pretrained_name"])

# Get predictions and probabilities
classes_2019, probs_2019 = classify_and_get_probs(patches_2019, model)
classes_2020, probs_2020 = classify_and_get_probs(patches_2020, model)

# Plot the distribution of classes for both years
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

# Use Otsu's method to find the optimal threshold for the Euclidean distance between the two years 
euclidean_scores = np.linalg.norm(probs_2020 - probs_2019, axis=1)
otsu_euclidean = threshold_otsu(euclidean_scores)
labels_euclidean = (euclidean_scores > otsu_euclidean).astype(np.uint8)
counts_euclidean = Counter(labels_euclidean)
plot_histogram(
    data=[counts_euclidean[0], counts_euclidean[1]],
    title="Healthy vs Damaged (distanza Euclidea + Otsu)",
    xlabel="",
    ylabel="Numero di patch",
    xticks_labels=["Healthy", "Damaged"],
    save_path=os.path.join(config.paths["results_dir"], "euclidean_otsu_damaged.png")
)

# Use Otsu's method to find the optimal threshold for the SAM distance between the two years
sam_scores = compute_sam(probs_2019, probs_2020)
otsu_sam = threshold_otsu(sam_scores)
labels_sam = (sam_scores > otsu_sam).astype(np.uint8)
counts_sam = Counter(labels_sam)
plot_histogram(
    data=[counts_sam[0], counts_sam[1]],
    title="Healthy vs Damaged (SAM + Otsu)",
    xlabel="",
    ylabel="Numero di patch",
    xticks_labels=["Healthy", "Damaged"],
    save_path=os.path.join(config.paths["results_dir"], "sam_otsu_damaged.png")
)
