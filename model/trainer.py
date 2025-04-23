import torch
from tqdm import tqdm
import os
from datetime import datetime
from model.utils import compute_epoch_metrics, log_epoch_results, build_optimizer, build_scheduler
from model.model import Pretrainedmodel

class Trainer:
    """Trainer class to handle training, validation, testing, and full training loop."""

    def __init__(self, config):
        """Initialize the Trainer class and set up loss, optimizer, scheduler."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Pretrainedmodel.from_pretrained(
            config.model["pretrained_name"],
            num_classes=2
        ).to(self.device)
        self.experiment_name = config.training["experiment_name"]
        self.results_path = os.path.join(config.paths["results_dir"], config.filenames["metrics_results"])
        os.makedirs(config.paths["results_dir"], exist_ok=True)

        class_weights = torch.tensor(config.training["class_weights"]).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = build_optimizer(config, self.model.parameters())
        self.scheduler = build_scheduler(config, self.optimizer, config.training["batch_size"], 0)

    def run_epoch(self, loader, split):
        """Run a single epoch for training, validation or testing."""

        # Set model mode
        self.model.train() if split == "train" else self.model.eval()

        total_loss = 0.0
        all_labels, all_preds = [], []
        start_time = datetime.now()

        # Progress bar
        loop = tqdm(loader, total=len(loader), desc=split.capitalize(), leave=False)
        for images, labels, _, _ in loop:
            images, labels = images.to(self.device), labels.to(self.device)

            if split == "train":
                self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            if split == "train":
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

            total_loss += loss.item()
            loop.set_postfix({'Batch Loss': loss.item()})

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)

        # Compute metrics
        elapsed_time = str(datetime.now() - start_time).split(".")[0]
        metrics = compute_epoch_metrics(all_labels, all_preds, avg_loss, elapsed_time)
        return metrics

    def train_epoch(self, loader):
        """Train the model for one epoch"""
        return self.run_epoch(loader, split="train")

    @torch.no_grad()
    def validate_epoch(self, loader):
        """Validate the model for one epoch"""
        return self.run_epoch(loader, split="val")

    @torch.no_grad()
    def test_epoch(self, loader):
        """Test the model for one epoch"""
        return self.run_epoch(loader, split="test")

    def train_model(self, train_loader, val_loader, test_loader, config):
        """Train the model with early stopping and model saving."""

        # Training parameters
        patience = config.training["patience"]
        best_model_path = os.path.join(config.paths["results_dir"], config.training["experiment_name"] + ".pth")
        best_f1_val = 0.0
        no_improve_epochs = 0

        # Training loop
        now = datetime.now()

        for epoch in range(config.training["epochs"]):
            print(f"\nEpoch [{epoch+1}/{config.training['epochs']}] starting...")

            # Train and validate
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            test_metrics = self.test_epoch(test_loader)

            log_epoch_results(
                epoch + 1,
                train_metrics,
                val_metrics,
                test_metrics,
                config.training["experiment_name"],
                self.results_path
            )

            # Early stopping and model saving
            if val_metrics["F1_1"] > best_f1_val:
                best_f1_val = val_metrics["F1_1"]
                no_improve_epochs = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

        print(f"\nTraining complete. Total time: {str(datetime.now() - now).split('.')[0]}")