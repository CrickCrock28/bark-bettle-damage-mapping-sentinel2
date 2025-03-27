import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import csv
import os

class Trainer:
    """Trainer class to handle training and validation of the model."""
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

    def run_epoch(self, loader, epoch, training):
        """Run a single epoch of training, validation or testing."""
        
        # Set model to the correct mode
        self.model.train() if training else self.model.eval()
        
        total_loss = 0.0
        all_labels, all_probs = [], []

        # Iterate over the data
        loop = tqdm(loader, total=len(loader), desc="Training" if training else "Validating", leave=False)
        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)
            
            if training:
                self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            if training:
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

            total_loss += loss.item()
            loop.set_postfix({'Batch Loss': loss.item()})

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().astype(int).flatten())

        avg_loss = total_loss / len(loader)

        # Compute metrics
        all_preds = (torch.tensor(all_probs) > 0.5).int().numpy()
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

        # Write metrics to CSV
        split = "train" if training else "val"
        csv_path = os.path.join(self.log_dir, self.metrics_filename_format.format(split=split))
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['Epoch', 'Loss', 'Precision', 'Recall', 'F1', 'TN', 'FP', 'FN', 'TP'])
            writer.writerow([epoch, avg_loss, precision, recall, f1, tn, fp, fn, tp])

        return avg_loss

    def train_epoch(self, loader, epoch):
        """Train the model for one epoch"""
        return self.run_epoch(loader, epoch, training=True)

    @torch.no_grad()
    def validate_epoch(self, loader, epoch):
        """Validate the model for one epoch"""
        return self.run_epoch(loader, epoch, training=False)

