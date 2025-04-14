import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from src.pretrain.reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
import warnings

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
 # FIXME are them correct?
from tqdm import tqdm

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


# Load the preprocessed data
# FIXME hardcoded
data_2019 = np.load("data/preprocessed/2019-09-01 - 2019-09-30/test/filtered.npz")
data_2020 = np.load("data/preprocessed/2020-09-01 - 2020-09-30/test/filtered.npz")

patches_2019 = data_2019["patches"]
patches_2020 = data_2020["patches"]

if patches_2019.shape != patches_2020.shape:
    raise ValueError(f"Shape mismatch: patches from 2019 have shape {patches_2019.shape}, while patches from 2020 have shape {patches_2020.shape}")

# Load the pre-trained model
# FIXME hardcoded
model = BigEarthNetv2_0_ImageClassifier.from_pretrained("BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0")

# Classify the patches
classes_2019 = classify_patches(patches_2019, model)
classes_2020 = classify_patches(patches_2020, model)

# Get the changed patches counts
change_labels = (classes_2019 != classes_2020).astype(np.uint8)  # 1 = changed, 0 = unchanged
changed_counts = Counter(change_labels)

# Histogram for the distribution of classes in 2019
plt.figure(figsize=(12, 8))
class_counts = Counter(classes_2019)
# plt.bar(range(19), [class_counts[i] for i in range(19)], tick_label=[i for i in range(19)])
plt.bar(range(19), [class_counts[i] for i in range(19)], tick_label=[class_names[i] for i in range(19)])
plt.xticks(rotation=45, ha="right")
plt.title("Distribution of classes (2019)")
plt.xlabel("Class")
plt.ylabel("Number of patches")
plt.tight_layout()
plt.savefig("class_distribution_2019.png")
plt.show()

# Histogram for the distribution of classes in 2020
plt.figure(figsize=(12, 8))
class_counts = Counter(classes_2020)
# plt.bar(range(19), [class_counts[i] for i in range(19)], tick_label=[i for i in range(19)])
plt.bar(range(19), [class_counts[i] for i in range(19)], tick_label=[class_names[i] for i in range(19)])
plt.xticks(rotation=45, ha="right")
plt.title("Distribution of classes (2020)")
plt.xlabel("Class")
plt.ylabel("Number of patches")
plt.tight_layout()
plt.savefig("class_distribution_2020.png")
plt.show()

# Histogram: changed vs unchanged patches
plt.figure(figsize=(5, 4))
plt.bar(["Healthy", "Damaged"], [changed_counts[0], changed_counts[1]], color=["green", "red"])
plt.title("Patch Damaged vs Healthy")
plt.ylabel("Number of patches")
plt.tight_layout()
plt.savefig("damage_vs_Healthy.png")
plt.show()
