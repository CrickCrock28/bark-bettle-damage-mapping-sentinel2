import torch
from torch.utils.data import DataLoader
from data.dataset_preprocessed import NPZSentinelDataset
from model.model import Pretrainedmodel
from config.config_loader import Config
from model.trainer import Trainer
from model.utils import freeze_backbone, unfreeze_backbone, build_optimizer, build_scheduler, calculate_positive_weight
import warnings
import gc
from datetime import datetime

#FIXME workaround for the waning message
warnings.filterwarnings("ignore", message="Keyword 'img_size' unknown*")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load configuration
    config = Config()

    # Load datasets and data loaders
    target_size = tuple(config.dataset["target_size"])
    train_data_dir = config.paths["preprocessed_train_dir"]
    test_data_dir = config.paths["preprocessed_test_dir"]

    train_dataset = NPZSentinelDataset(
        data_dir=train_data_dir,
        target_size=target_size,
        resize_mode=config.dataset["resize_mode"]
    )

    test_dataset = NPZSentinelDataset(
        data_dir=test_data_dir,
        target_size=target_size,
        resize_mode=config.dataset["resize_mode"]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.training["batch_size"],
        shuffle=True,
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

    # Load model
    model = Pretrainedmodel.from_pretrained(
        config.model["pretrained_name"],
        num_classes=1
    ).to(device)

    freeze_backbone(model)

    # Load loss function, optimizer and scheduler
    pos_weight = calculate_positive_weight(train_data_dir, device, config.filenames["block"])
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
        log_dir=config.paths["log_dir"]
    )

    epochs = config.training["epochs"]
    warmup_epochs = config.training["warmup_epochs"]

    try:
        total_start = datetime.now()
        for epoch in range(epochs):
            start_epoch = datetime.now()
            print(f"\nEpoch [{epoch+1}/{epochs}] starting...")

            if epoch == warmup_epochs:
                unfreeze_backbone(model)
                optimizer = build_optimizer(config, model.parameters())
                trainer.optimizer = optimizer
                scheduler = build_scheduler(config, optimizer, len(train_loader), epoch)
                trainer.scheduler = scheduler

            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate_epoch(test_loader)

            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            trainer.evaluate_and_log(test_loader, epoch=epoch+1)
            print(f"===> Epoch [{epoch+1}] completed.")
            time_taken = datetime.now() - start_epoch
            print(f"Time taken: {str(time_taken).split('.')[0]}")

        total_time_taken = datetime.now() - total_start
        print(f"Training completed. Total time taken: {str(total_time_taken).split('.')[0]}")

    finally:
        train_dataset.clear_memory()
        test_dataset.clear_memory()
        gc.collect()

if __name__ == "__main__":
    main()
