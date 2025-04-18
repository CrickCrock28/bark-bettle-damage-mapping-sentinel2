import torch
from tqdm import tqdm
import os
from datetime import timedelta
import time
from model.utils import compute_metrics

class Trainer:
    """Trainer class to handle training, validation and testing of the model."""

    def __init__(self, model, criterion, optimizer, device, results_dir, results_filename, experiment_name, scheduler=None):
        """Initialize the Trainer class"""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.results_path = os.path.join(results_dir, results_filename)
        self.experiment_name = experiment_name
        os.makedirs(results_dir, exist_ok=True)

    def run_epoch(self, loader, split):
        """Run a single epoch for training, validation or testing."""
        
        # Set model mode
        self.model.train() if split == "train" else self.model.eval()
        
        total_loss = 0.0
        all_labels, all_preds = [], []
        start_time = time.time()

        # Progress bar
        loop = tqdm(loader, total=len(loader), desc=split.capitalize(), leave=False)
        for images, labels in loop:
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
        elapsed_time = str(timedelta(seconds=time.time() - start_time))
        metrics = compute_metrics(all_labels, all_preds, avg_loss, elapsed_time)
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
