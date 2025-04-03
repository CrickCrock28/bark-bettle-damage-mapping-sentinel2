import torch
from torch.utils.data import DataLoader
from data.dataset_preprocessed import NPZSentinelDataset
from model.model import Pretrainedmodel
from config.config_loader import Config
from model.trainer import Trainer
from model.utils import build_optimizer, build_scheduler
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
        resize_mode=config.dataset["resize_mode"],
        preprocessed_filename=config.filenames["preprocessed"]
    )

    # Split full training set into train and validation sets
    total_train_len = len(full_train_dataset)
    train_len = int(0.8 * total_train_len)
    train_dataset = torch.utils.data.Subset(full_train_dataset, range(0, train_len))
    val_dataset = torch.utils.data.Subset(full_train_dataset, range(train_len, total_train_len))

    # Load full test dataset and split into validation and test sets
    test_dataset = NPZSentinelDataset(
        data_dir=test_data_dir,
        target_size=target_size,
        resize_mode=config.dataset["resize_mode"],
        preprocessed_filename=config.filenames["preprocessed"]
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
        num_classes=2
    ).to(device)

    # Loss function, optimizer, scheduler
    class_weights = torch.tensor([config.training["class_weights"][0], config.training["class_weights"][1]]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
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
    patience = config.training["patience"]
    best_model_path = os.path.join(config.paths["log_dir"], config.filenames["best_model"])
    best_f1_val = 0.0
    best_f1_test = 0.0
    no_improve_epochs = 0

    # Training loop
    try:
        total_start = datetime.now()
        # Iterate over epochs
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}] starting...")

            # Train and validate
            train_f1 = trainer.train_epoch(train_loader, epoch+1)
            val_f1 = trainer.validate_epoch(val_loader, epoch+1)
            test_f1 = trainer.test_epoch(test_loader, epoch+1)

            # Save best model
            if test_f1 > best_f1_test:
                best_f1_test = test_f1
                torch.save(model.state_dict(), best_model_path)

            # Early stopping
            if val_f1 > best_f1_val:
                best_f1_val = val_f1
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

            print(f"===> Epoch [{epoch+1}] completed.")

        total_time = datetime.now() - total_start
        print(f"\nTraining complete. Total time: {str(total_time).split('.')[0]}")

    finally:
        # Clear memory
        full_train_dataset.clear_memory()
        test_dataset.clear_memory()
        gc.collect()

if __name__ == "__main__":
    main()
