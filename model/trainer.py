import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
import os

class Trainer:
    def __init__(self, model, criterion, optimizer, device, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False)

        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Batch Loss': loss.item()})

        avg_loss = total_loss / len(loader)
        return avg_loss

    @torch.no_grad()
    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(loader, total=len(loader), desc="Validating", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            progress_bar.set_postfix({'Batch Loss': loss.item()})

        avg_loss = total_loss / len(loader)
        return avg_loss

    @torch.no_grad()
    def evaluate_metrics(self, loader, dynamic_threshold=False, epoch=None, log_dir="logs"):
        self.model.eval()
        all_labels = []
        all_probs = []
        progress_bar = tqdm(loader, total=len(loader), desc="Evaluating", leave=False)

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.cpu().numpy().astype(int)

            all_labels.extend(labels.flatten())
            all_probs.extend(probs.flatten())

        if dynamic_threshold:
            precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
            f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
            best_idx = f1_scores.argmax()
            best_threshold = thresholds[best_idx]
            print(f"Dynamic Threshold selected: {best_threshold:.2f}")
        else:
            best_threshold = 0.3

        all_preds = (torch.tensor(all_probs) > best_threshold).int().numpy()

        # Metrics
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"], output_dict=True, zero_division=0)
        recall = recall_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print("Confusion Matrix:")
        print(cm)
        print(f"===> Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        os.makedirs(log_dir, exist_ok=True)
        csv_file = os.path.join(log_dir, "metrics.csv")
        fieldnames = ['Epoch', 'Threshold', 'Precision', 'Recall', 'F1', 
                    'TN', 'FP', 'FN', 'TP']

        tn, fp, fn, tp = cm.ravel()

        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'Epoch': epoch,
                'Threshold': round(best_threshold, 4),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1': round(f1, 4),
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TP': tp
            })

        return best_threshold
