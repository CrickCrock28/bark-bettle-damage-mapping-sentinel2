import torch
from torch.utils.data import DataLoader, random_split
from data.dataset_preprocessed import NPZSentinelDataset
from model.model import Pretrainedmodel
from config.config_loader import Config
from model.trainer import Trainer
from model.utils import freeze_backbone, unfreeze_backbone, build_optimizer, build_scheduler, calculate_positive_weight
import warnings
import gc
from datetime import datetime
import os

# FIXME workaround for the warning message
warnings.filterwarnings("ignore", message="Keyword 'img_size' unknown*")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load configuration
    config = Config()

    # Load datasets
    target_size = tuple(config.dataset["target_size"])
    train_data_dir = config.paths["preprocessed_train_dir"]
    test_data_dir = config.paths["preprocessed_test_dir"]

    # Load train dataset
    full_train_dataset = NPZSentinelDataset(
        data_dir=train_data_dir,
        target_size=target_size,
        resize_mode=config.dataset["resize_mode"]
    )
    
    # Split full training set into train and validation sets
    total_train_len = len(full_train_dataset)
    train_len = int(0.8 * total_train_len)
    val_len = total_train_len - train_len
    train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len])

    # Load full test dataset and split into validation and test sets
    test_dataset = NPZSentinelDataset(
        data_dir=test_data_dir,
        target_size=target_size,
        resize_mode=config.dataset["resize_mode"]
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.training["batch_size"],
        shuffle=True,
        num_workers=config.dataset["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.training["batch_size"],
        shuffle=False,
        num_workers=config.dataset["num_workers"],
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training["batch_size"],
        shuffle=False,
        num_workers=config.dataset["num_workers"],
        pin_memory=True
    )

    # Load and freeze the model
    model = Pretrainedmodel.from_pretrained(
        config.model["pretrained_name"],
        num_classes=1
    ).to(device)
    freeze_backbone(model)

    # Loss function, optimizer, scheduler
    # pos_weight = calculate_positive_weight(train_data_dir, device, config.filenames["block"])
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer, len(train_loader), 0)

    # Trainer setup
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        log_dir=config.paths["log_dir"],
        metrics_filename_format=config.filenames["metrics"]
    )

    # Training parameters
    epochs = config.training["epochs"]
    warmup_epochs = config.training["warmup_epochs"]
    best_val_loss = float("inf")
    best_model_path = os.path.join(config.paths["log_dir"], config.filenames["best_model"])

    # Training loop
    try:
        total_start = datetime.now()
        # Iterate over epochs
        for epoch in range(epochs):
            start_epoch = datetime.now()
            print(f"\nEpoch [{epoch+1}/{epochs}] starting...")

            # Unfreeze backbone after warmup epochs
            if epoch == warmup_epochs:
                unfreeze_backbone(model)
                optimizer = build_optimizer(config, model.parameters())
                trainer.optimizer = optimizer
                scheduler = build_scheduler(config, optimizer, len(train_loader), epoch)
                trainer.scheduler = scheduler

            # Train and validate
            train_loss = trainer.train_epoch(train_loader, epoch)
            val_loss = trainer.validate_epoch(val_loader, epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"===> Epoch [{epoch+1}] completed. Time: {str(datetime.now() - start_epoch).split('.')[0]}")

        total_time = datetime.now() - total_start
        print(f"\nTraining complete. Total time: {str(total_time).split('.')[0]}")

        # Load best model and evaluate on test set
        print("\nEvaluating best model on test set...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        trainer.validate_epoch(test_loader, epoch="final_test")

    finally:
        # Clear memory
        full_train_dataset.clear_memory()
        test_dataset.clear_memory()
        gc.collect()

if __name__ == "__main__":
    main()
