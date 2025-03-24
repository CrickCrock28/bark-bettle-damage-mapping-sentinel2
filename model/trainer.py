import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import csv
import os

class Trainer:
    """Trainer class to handle training and validation of the model."""
    def __init__(self, model, criterion, optimizer, device, log_dir, scheduler=None):
        """Initialize the Trainer class"""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def run_epoch(self, loader, training=True):
        """Run a single epoch of training or validation"""
        self.model.train() if training else self.model.eval()
        total_loss = 0.0

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

        avg_loss = total_loss / len(loader)
        return avg_loss

    def train_epoch(self, loader):
        """Train the model for one epoch"""
        return self.run_epoch(loader, training=True)

    @torch.no_grad()
    def validate_epoch(self, loader):
        """Validate the model for one epoch"""
        return self.run_epoch(loader, training=False)

    @torch.no_grad()
    def compute_metrics(self, loader):
        """Compute metrics for the model on the given dataset"""
        self.model.eval()
        all_labels, all_probs = [], []

        # Iterate over the dataset to get predictions and labels
        for images, labels in tqdm(loader, total=len(loader), desc="Evaluating", leave=False):
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().astype(int).flatten())

        # Compute metrics
        all_preds = (torch.tensor(all_probs) > 0.5).int().numpy()

        cm = confusion_matrix(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        # Print metrics
        print("Confusion Matrix:")
        print(cm)
        print(f"===> Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        metrics = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'confusion_matrix': cm
        }

        return metrics

    def save_metrics(self, metrics, epoch):
        """Save metrics to a CSV file"""
        csv_file = os.path.join(self.log_dir, "metrics.csv")
        fieldnames = ['Epoch', 'Precision', 'Recall', 'F1', 'TN', 'FP', 'FN', 'TP']
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'Epoch': epoch,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TP': tp
            })

    def evaluate_and_log(self, loader, epoch):
        """Evaluate the model on the given dataset and log the metrics"""
        metrics = self.compute_metrics(loader)
        self.save_metrics(metrics, epoch)
