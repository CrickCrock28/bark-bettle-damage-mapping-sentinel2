import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SentinelDataset, load_data
from model import Pretrainedmodel

from config.config_loader import Config

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    return total_loss / len(loader)

def main():

    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_channels = config.channels["current"]
    selected_channels = config.channels["selected"]
    channels_order = [current_channels.index(c)+1 for c in selected_channels]

    train_images, train_masks = load_data(
        config.paths["train_mask_dir"],
        config.paths["sentinel_data_dirs"],
        config.filenames["masks"],
        config.filenames["images"]
    )

    test_images, test_masks = load_data(
        config.paths["test_mask_dir"],
        config.paths["sentinel_data_dirs"],
        config.filenames["masks"],
        config.filenames["images"]
    )

    train_dataset = SentinelDataset(
        train_images,
        train_masks,
        channels_order
    )
    test_dataset = SentinelDataset(
        test_images,
        test_masks,
        channels_order
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.training["batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training["batch_size"],
        shuffle=False
    )

    model = Pretrainedmodel.from_pretrained(
        config.model["pretrained_name"],
        num_classes=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=config.training["learning_rate"]
    )

    epochs = config.training["epochs"]
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, test_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")

if __name__ == "__main__":
    main()
