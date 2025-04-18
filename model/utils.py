from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import os

def build_optimizer(config, model_params):
    """Build optimizer based on configuration."""
    optimizer_type = config.training["optimizer"]["type"]
    optimizer_params = config.training["optimizer"]["params"]
    if optimizer_type == "Adam":
        return Adam(model_params, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def build_scheduler(config, optimizer, train_loader_length, current_epoch):
    """Build scheduler based on configuration."""
    scheduler_type = config.training["scheduler"]["type"]
    T_max_factor = config.training["scheduler"]["T_max_factor"]
    if scheduler_type == "CosineAnnealingLR":
        T_max = int(T_max_factor * (config.training["epochs"] - current_epoch) * train_loader_length)
        return CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

def compute_metrics(all_labels, all_preds, loss, time):
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

def log_epoch_results(epoch, train_metrics, val_metrics, test_metrics, experiment_name, results_path):
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
    if os.path.exists(results_path):
        try:
            # Try to append to the existing sheet
            existing_df = pd.read_excel(results_path, sheet_name=experiment_name)
            full_df = pd.concat([existing_df, df_new], ignore_index=True)
        except ValueError:
            # The sheet does not exist, create it
            full_df = df_new
        # Write to the existing file
        with pd.ExcelWriter(results_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            full_df.to_excel(writer, sheet_name=experiment_name, index=False)
    else:
        # The file does not exist, create it
        with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
            df_new.to_excel(writer, sheet_name=experiment_name, index=False)
