import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.dataset import SentinelDataset, load_data
from model.model import Pretrainedmodel
from config.config_loader import Config
from model.trainer import Trainer
from model.utils import freeze_backbone, unfreeze_backbone
from model.utils import FocalLoss

# FIXME workaround for the warning
import warnings
warnings.filterwarnings("ignore", message="Keyword 'img_size' unknown*")

def calculate_pos_weight(dataset, device):
    label_counts = dataset.get_labels_counts()
    if label_counts[1] > 0:
        pos_weight = torch.tensor([label_counts[0] / label_counts[1]], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device)
    return pos_weight

def build_optimizer(config, model_params):
    optimizer_type = config.optimizer["type"]
    optimizer_params = config.optimizer["params"]
    if optimizer_type == "Adam":
        return Adam(model_params, **optimizer_params)
    elif optimizer_type == "SGD":
        return SGD(model_params, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def build_scheduler(config, optimizer, train_loader_length, current_epoch, total_epochs):
    scheduler_type = config.training["scheduler"]["type"]
    T_max_factor = config.training["scheduler"].get("T_max_factor", 1)

    if scheduler_type == "CosineAnnealingLR":
        T_max = int(T_max_factor * (total_epochs - current_epoch) * train_loader_length)
        return CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

def main():
    config = Config()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Channel setup
    current_channels = config.channels["current"]
    selected_channels = config.channels["selected"]
    channels_order = [current_channels.index(c)+1 for c in selected_channels]

    # Load paths
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

    # Dataset setup
    target_size = tuple(config.dataset["target_size"])
    radius = config.dataset["radius"]
    sampling_rate = config.dataset["sampling_rate"]

    train_dataset = SentinelDataset(
        train_images,
        train_masks,
        channels_order,
        target_size=target_size,
        radius=radius,
        sampling_rate=sampling_rate
    )
    test_dataset = SentinelDataset(
        test_images,
        test_masks,
        channels_order,
        target_size=target_size,
        radius=radius,
        sampling_rate=sampling_rate
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.training["batch_size"],
        shuffle=True,
        num_workers=os.cpu_count()-1,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training["batch_size"],
        shuffle=False,
        num_workers=os.cpu_count()-1,
        pin_memory=True
    )

    # Model setup
    model = Pretrainedmodel.from_pretrained(
        config.model["pretrained_name"],
        num_classes=1
    ).to(device)

    freeze_backbone(model)

    pos_weight = calculate_pos_weight(train_dataset, device)
    criterion = FocalLoss(pos_weight=pos_weight, gamma=2)

    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer, len(train_loader), 0, config.training["epochs"])

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    dynamic_threshold = config.training["dynamic_threshold"]
    epochs = config.training["epochs"]
    warmup_epochs = config.training["warmup_epochs"]

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}] starting...")

        if epoch == warmup_epochs:
            unfreeze_backbone(model)
            optimizer = build_optimizer(config, model.parameters())
            trainer.optimizer = optimizer
            scheduler = build_scheduler(config, optimizer, len(train_loader), epoch, config.training["epochs"])
            trainer.scheduler = scheduler

        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate_epoch(test_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        threshold = trainer.evaluate_metrics(test_loader, dynamic_threshold=dynamic_threshold, epoch=epoch+1)
        print(f"===> Epoch [{epoch+1}] Threshold used: {threshold:.2f}")

if __name__ == "__main__":
    main()
