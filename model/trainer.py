import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import os
from datetime import timedelta
import time

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

    def compute_metrics(self, all_labels, all_preds, loss, time):
        """Compute metrics from true and predicted labels"""
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        precision = precision_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1])
        recall = recall_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1])
        f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1])
        return {
            'Loss': loss,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'Precision_0': precision[0], 'Recall_0': recall[0], 'F1_0': f1[0],
            'Precision_1': precision[1], 'Recall_1': recall[1], 'F1_1': f1[1],
            'Time': time,
        }

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
        metrics = self.compute_metrics(all_labels, all_preds, avg_loss, elapsed_time)
        return metrics

    def log_epoch_results(self, epoch, train_metrics, val_metrics, test_metrics, experiment_name):
        """Log metrics for current epoch into an Excel sheet"""
        
        # Create row with metrics
        row = {
            'Epoch': epoch,
            # Train
            'Train_Loss': f"{train_metrics['Loss']:.4f}",
            'Train_Precision_1': f"{train_metrics['Precision_1']:.4f}", 'Train_Recall_1': f"{train_metrics['Recall_1']:.4f}", 'Train_F1_1': f"{train_metrics['F1_1']:.4f}",
            'Train_Precision_0': f"{train_metrics['Precision_0']:.4f}", 'Train_Recall_0': f"{train_metrics['Recall_0']:.4f}", 'Train_F1_0': f"{train_metrics['F1_0']:.4f}", 'Train_TN': f"{train_metrics['TN']:.4f}", 'Train_FP': f"{train_metrics['FP']:.4f}", 'Train_FN': f"{train_metrics['FN']:.4f}", 'Train_TP': f"{train_metrics['TP']:.4f}",
            'Train_Time': train_metrics['Time'].split(".")[0],
            # Val
            'Val_Loss': f"{val_metrics['Loss']:.4f}",
            'Val_Precision_1': f"{val_metrics['Precision_1']:.4f}", 'Val_Recall_1': f"{val_metrics['Recall_1']:.4f}", 'Val_F1_1': f"{val_metrics['F1_1']:.4f}",
            'Val_Precision_0': f"{val_metrics['Precision_0']:.4f}", 'Val_Recall_0': f"{val_metrics['Recall_0']:.4f}", 'Val_F1_0': f"{val_metrics['F1_0']:.4f}", 'Val_TN': f"{val_metrics['TN']:.4f}", 'Val_FP': f"{val_metrics['FP']:.4f}", 'Val_FN': f"{val_metrics['FN']:.4f}", 'Val_TP': f"{val_metrics['TP']:.4f}",
            'Val_Time': val_metrics['Time'].split(".")[0],
            # Test
            'Test_Loss': f"{test_metrics['Loss']:.4f}",
            'Test_Precision_1': f"{test_metrics['Precision_1']:.4f}", 'Test_Recall_1': f"{test_metrics['Recall_1']:.4f}", 'Test_F1_1': f"{test_metrics['F1_1']:.4f}",
            'Test_Precision_0': f"{test_metrics['Precision_0']:.4f}", 'Test_Recall_0': f"{test_metrics['Recall_0']:.4f}", 'Test_F1_0': f"{test_metrics['F1_0']:.4f}", 'Test_TN': f"{test_metrics['TN']:.4f}", 'Test_FP': f"{test_metrics['FP']:.4f}", 'Test_FN': f"{test_metrics['FN']:.4f}", 'Test_TP': f"{test_metrics['TP']:.4f}",
            'Test_Time': test_metrics['Time'].split(".")[0],
        }

        # Create DataFrame and append to Excel file
        df_new = pd.DataFrame([row])
        if os.path.exists(self.results_path):
            try:
                # Try to append to the existing sheet
                existing_df = pd.read_excel(self.results_path, sheet_name=experiment_name)
                full_df = pd.concat([existing_df, df_new], ignore_index=True)
            except ValueError:
                # The sheet does not exist, create it
                full_df = df_new
            # Write to the existing file
            with pd.ExcelWriter(self.results_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                full_df.to_excel(writer, sheet_name=experiment_name, index=False)
        else:
            # The file does not exist, create it
            with pd.ExcelWriter(self.results_path, engine='openpyxl') as writer:
                df_new.to_excel(writer, sheet_name=experiment_name, index=False)

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
