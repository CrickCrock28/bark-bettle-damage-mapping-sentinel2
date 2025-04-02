import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import csv
import os
from datetime import timedelta
import time

class Trainer:
    """Trainer class to handle training, validation and testing of the model."""

    def __init__(self, model, criterion, optimizer, device, log_dir, metrics_filename_format, scheduler=None):
        """Initialize the Trainer class"""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.log_dir = log_dir
        self.metrics_filename_format = metrics_filename_format
        os.makedirs(self.log_dir, exist_ok=True)

    def run_epoch(self, loader, epoch, split):
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
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0, labels=[0,1])
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0, labels=[0,1])
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=[0,1])

        # Write metrics to CSV
        csv_path = os.path.join(self.log_dir, self.metrics_filename_format)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'Epoch', 'Split', 'Loss',
                    'Precision_1', 'Recall_1', 'F1_1',
                    'Precision_0', 'Recall_0', 'F1_0',
                    'TN', 'FP', 'FN', 'TP',
                    'Time'
                ])
            writer.writerow([
                epoch, split, f"{avg_loss:.4f}",                
                f"{precision_per_class[1]:.4f}", f"{recall_per_class[1]:.4f}", f"{f1_per_class[1]:.4f}",
                f"{precision_per_class[0]:.4f}", f"{recall_per_class[0]:.4f}", f"{f1_per_class[0]:.4f}",
                tn, fp, fn, tp,
                f"{timedelta(seconds=time.time() - start_time)}"
            ])

        return f1_per_class[1]

    def train_epoch(self, loader, epoch):
        """Train the model for one epoch"""
        return self.run_epoch(loader, epoch, split="train")

    @torch.no_grad()
    def validate_epoch(self, loader, epoch):
        """Validate the model for one epoch"""
        return self.run_epoch(loader, epoch, split="val")

    @torch.no_grad()
    def test_epoch(self, loader, epoch):
        """Test the model for one epoch"""
        return self.run_epoch(loader, epoch, split="test")
