import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from .models import EGCN, GCN, GAT, GraphSAGE, GIN, ChebNet, FocalLoss

def choose_model(model_type, in_feats, hidden_feats, out_feats, num_layers=2, num_heads=1, k=3, feat_drop=0.0, attn_drop=0.0, dropout=0.0, do_train=True):
    if model_type == 'GraphSAGE':
        return GraphSAGE(in_feats, hidden_feats, out_feats)
    elif model_type == 'GAT':
        return GAT(in_feats, out_feats, num_layers=num_layers, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, do_train=do_train)
    elif model_type == 'GCN':
        return GCN(in_feats, out_feats, num_layers=num_layers, do_train=do_train)
    elif model_type == 'GIN':
        return GIN(in_feats, out_feats, num_layers=num_layers)
    elif model_type == 'EGCN':
        return EGCN(in_feats, hidden_feats, out_feats, k=k, dropout=dropout)
    elif model_type == 'ChebNet':
        return ChebNet(in_feats, hidden_feats, out_feats)
    else:
        raise ValueError("Invalid model type. Choose from ['GraphSAGE', 'GAT', 'GCN', 'GIN', 'ChebNet', 'EGCN'].")

def save_overall_metrics(total_time, average_time_per_epoch, average_cpu_usage, average_gpu_usage, args, output_dir):
    """
    Save the overall performance metrics to a CSV file.

    Args:
    - total_time: Total training time in seconds.
    - average_time_per_epoch: Average time per epoch in seconds.
    - average_cpu_usage: Average CPU usage in MB.
    - average_gpu_usage: Average GPU usage in MB.
    - args: Argument object containing model and network type.
    - output_dir: The directory where the results will be saved.
    """
    # Save overall metrics
    df_overall_metrics = pd.DataFrame([{
        "Model Type": args.model_type,
        "Total Time": f"{total_time:.4f}s",
        "Average Time per Epoch": f"{average_time_per_epoch:.4f}s",
        "Average CPU Usage (MB)": f"{average_cpu_usage:.2f}",
        "Average GPU Usage (MB)": f"{average_gpu_usage:.2f}"
    }])
    
    # Define path to save the CSV
    overall_metrics_csv_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_overall_performance_epo{args.epochs}_{args.in_feats}.csv')
    
    # Save to CSV
    df_overall_metrics.to_csv(overall_metrics_csv_path, index=False)
    print(f"Overall performance metrics saved to {overall_metrics_csv_path}")

def find_evidence(row,df2):
    # Condition 1: Match on miRNA == source and disease == destination
    match1 = df2[(df2['miRNA'] == row['source']) & (df2['disease'] == row['destination'])]
    
    # Condition 2: Match on miRNA == destination and disease == source
    match2 = df2[(df2['miRNA'] == row['destination']) & (df2['disease'] == row['source'])]

    if not match1.empty:
        return match1.iloc[0]['reference']  # Return the reference from the first match in match1
    elif not match2.empty:
        return match2.iloc[0]['reference']  # Return the reference from the first match in match2
    else:
        return 'Unconfirmed'  # Return 'Unconfirmed' if no match is found

def plot_roc_pr_curves(fold_results, output_path):
    plt.figure(figsize=(18, 6))

    # Subplot for ROC curves
    plt.subplot(1, 2, 1)
    # Define the mean FPR values for interpolation
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    for i, (true_labels, predicted_scores) in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
        roc_auc = auc(fpr, tpr)

        # Interpolate the TPR values at the common FPR values
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        mean_tpr += tpr_interp

        plt.plot(fpr, tpr, lw=1, alpha=1.0, label=f'Fold {i + 1} (AUC = {roc_auc:.4f})')

    # Average the TPR values across folds
    mean_tpr /= len(fold_results)
    mean_auc = auc(mean_fpr, mean_tpr)

    ##plt.plot([0, 1], [0, 1], lw=1, color='r', alpha=0.8, linestyle='--')
    plt.plot(mean_fpr, mean_tpr, color='cyan', label=f'Mean  (AUC = {mean_auc:.4f})', lw=1.0, alpha=1.0)
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(False)  # Remove grid

    # Subplot for PR curves
    plt.subplot(1, 2, 2)
    # Define the mean Recall values for interpolation
    mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)

    for i, (true_labels, predicted_scores) in enumerate(fold_results):
        precision, recall, _ = precision_recall_curve(true_labels, predicted_scores)
        pr_auc = average_precision_score(true_labels, predicted_scores)

        # Interpolate the Precision values at the common Recall values
        precision_interp = np.interp(mean_recall, recall[::-1], precision[::-1])
        mean_precision += precision_interp

        plt.plot(recall, precision, lw=1, alpha=1.0, label=f'Fold {i+1} (PR = {pr_auc:.4f})')

    # Average the Precision values across folds
    mean_precision /= len(fold_results)
    mean_auc = auc(mean_recall, mean_precision)

    ##plt.plot([0, 1], [1, 0], lw=1, color='r', alpha=0.8, linestyle='--')
    plt.plot(mean_recall, mean_precision, color='cyan', label=f'Mean  (PR = {mean_auc:.4f})', lw=1.0, alpha=1.0)


    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(loc='lower left', fontsize='small')
    plt.grid(False)  # Remove grid

    # Adjust space between the two subplots
    plt.subplots_adjust(wspace=0.4)  # Increase wspace value to increase the space between subplots

    ##plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_training_validation_metrics(
        train_accuracies, avg_train_accuracies,
        val_accuracies, avg_val_accuracies,
        train_losses, avg_train_losses,
        val_losses, avg_val_losses,
        output_path, args):
    """
    Plot training and validation metrics including accuracy and loss over epochs.

    Parameters:
    - train_accuracies: List of lists containing training accuracy values for each fold.
    - avg_train_accuracies: List of average training accuracy values over epochs.
    - val_accuracies: List of lists containing validation accuracy values for each fold.
    - avg_val_accuracies: List of average validation accuracy values over epochs.
    - train_losses: List of lists containing training loss values for each fold.
    - avg_train_losses: List of average training loss values over epochs.
    - val_losses: List of lists containing validation loss values for each fold.
    - avg_val_losses: List of average validation loss values over epochs.
    - output_path: Directory path to save the plot.
    - args: Arguments containing model parameters for filename.
    """
    plt.figure(figsize=(12, 8))

    # Plot training accuracy
    plt.subplot(2, 2, 1)
    for i, acc in enumerate(train_accuracies):
        plt.plot(acc, label=f'Fold {i + 1}', linewidth=1)
    plt.plot(avg_train_accuracies, color='cyan', label='Mean', linewidth=1, alpha=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend(fontsize='small')  # Make legend text smaller

    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    for i, acc in enumerate(val_accuracies):
        plt.plot(acc, label=f'Fold {i + 1}', linewidth=1)
    plt.plot(avg_val_accuracies, color='cyan', label='Mean', linewidth=1, alpha=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend(fontsize='small')  # Make legend text smaller

    # Plot training loss
    plt.subplot(2, 2, 3)
    for i, loss in enumerate(train_losses):
        plt.plot(loss, label=f'Fold {i + 1}', linewidth=1)
    plt.plot(avg_train_losses, color='cyan', label='Mean', linewidth=1, alpha=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend(fontsize='small')  # Make legend text smaller

    # Plot validation loss
    plt.subplot(2, 2, 4)
    for i, loss in enumerate(val_losses):
        plt.plot(loss, label=f'Fold {i + 1}', linewidth=1)
    plt.plot(avg_val_losses, color='cyan', label='Mean', linewidth=1, alpha=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend(fontsize='small')  # Make legend text smaller

    # Adjust the horizontal space between the plots
    plt.subplots_adjust(wspace=0.4)  # Increase the space between the columns (default is usually 0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{args.model_type}_{args.net_type}_train_val_metrics_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'))
    plt.close()


def output_path_plot_training_validation_metrics(
        train_accuracies, avg_train_accuracies,
        val_accuracies, avg_val_accuracies,
        train_losses, avg_train_losses,
        val_losses, avg_val_losses,
        output_path):
    """
    Plot training and validation metrics including accuracy and loss over epochs.

    Parameters:
    - train_accuracies: List of lists containing training accuracy values for each fold.
    - avg_train_accuracies: List of average training accuracy values over epochs.
    - val_accuracies: List of lists containing validation accuracy values for each fold.
    - avg_val_accuracies: List of average validation accuracy values over epochs.
    - train_losses: List of lists containing training loss values for each fold.
    - avg_train_losses: List of average training loss values over epochs.
    - val_losses: List of lists containing validation loss values for each fold.
    - avg_val_losses: List of average validation loss values over epochs.
    - output_path: Directory path to save the plot.
    - args: Arguments containing model parameters for filename.
    """
    plt.figure(figsize=(12, 8))

    # Plot training accuracy
    plt.subplot(2, 2, 1)
    for i, acc in enumerate(train_accuracies):
        plt.plot(acc, label=f'Train Fold {i + 1}', linewidth=1)
    plt.plot(avg_train_accuracies, color='blue', label='Mean', linewidth=1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend(fontsize='small')  # Make legend text smaller

    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    for i, acc in enumerate(val_accuracies):
        plt.plot(acc, label=f'Val Fold {i + 1}', linewidth=1)
    plt.plot(avg_val_accuracies, color='blue', label='Mean', linewidth=1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend(fontsize='small')  # Make legend text smaller

    # Plot training loss
    plt.subplot(2, 2, 3)
    for i, loss in enumerate(train_losses):
        plt.plot(loss, label=f'Fold {i + 1}', linewidth=1)
    plt.plot(avg_train_losses, color='blue', label='Mean', linewidth=1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend(fontsize='small')  # Make legend text smaller

    # Plot validation loss
    plt.subplot(2, 2, 4)
    for i, loss in enumerate(val_losses):
        plt.plot(loss, label=f'Fold {i + 1}', linewidth=1)
    plt.plot(avg_val_losses, color='blue', label='Mean', linewidth=1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend(fontsize='small')  # Make legend text smaller

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def average_line_plot_training_validation_metrics(
        train_accuracies, avg_train_accuracies,
        val_accuracies, avg_val_accuracies,
        train_losses, avg_train_losses,
        val_losses, avg_val_losses,
        output_path, args):
    """
    Plot training and validation metrics including accuracy and loss over epochs.

    Parameters:
    - train_accuracies: List of lists containing training accuracy values for each fold.
    - avg_train_accuracies: List of average training accuracy values over epochs.
    - val_accuracies: List of lists containing validation accuracy values for each fold.
    - avg_val_accuracies: List of average validation accuracy values over epochs.
    - train_losses: List of lists containing training loss values for each fold.
    - avg_train_losses: List of average training loss values over epochs.
    - val_losses: List of lists containing validation loss values for each fold.
    - avg_val_losses: List of average validation loss values over epochs.
    - output_path: Directory path to save the plot.
    - args: Arguments containing model parameters for filename.
    """
    plt.figure(figsize=(12, 8))

    # Plot training accuracy
    plt.subplot(2, 2, 1)
    for i, acc in enumerate(train_accuracies):
        plt.plot(acc, label=f'Train Fold {i + 1}', linewidth=1)
    plt.plot(avg_train_accuracies, label='Average Train Accuracy', linewidth=1.5)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend(fontsize='small')  # Make legend text smaller

    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    for i, acc in enumerate(val_accuracies):
        plt.plot(acc, label=f'Val Fold {i + 1}', linewidth=1)
    plt.plot(avg_val_accuracies, label='Average Val Accuracy', linewidth=1.5)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend(fontsize='small')  # Make legend text smaller

    # Plot training loss
    plt.subplot(2, 2, 3)
    for i, loss in enumerate(train_losses):
        plt.plot(loss, label=f'Train Fold {i + 1}', linewidth=1)
    plt.plot(avg_train_losses, label='Average Train Loss', linewidth=1.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend(fontsize='small')  # Make legend text smaller

    # Plot validation loss
    plt.subplot(2, 2, 4)
    for i, loss in enumerate(val_losses):
        plt.plot(loss, label=f'Val Fold {i + 1}', linewidth=1)
    plt.plot(avg_val_losses, label='Average Val Loss', linewidth=1.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Epochs')
    plt.legend(fontsize='small')  # Make legend text smaller

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'train_val_metrics_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'))
    plt.show()

def plot_validation_accuracy(accuracies, avg_accuracies, output_path, args):
    """
    Plot validation accuracy over epochs for each fold and the average accuracy.
    
    Parameters:
    - accuracies: List of lists containing accuracy values for each fold.
    - avg_accuracies: List of average accuracy values over epochs.
    - output_path: Directory path to save the plot.
    - args: Arguments containing model parameters for filename.
    """
    plt.figure(figsize=(10, 6))
    for i, acc in enumerate(accuracies):
        plt.plot(acc, label=f'Fold {i + 1}')
    plt.plot(avg_accuracies, color='blue', label='Average Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_path, f'{args.model_type}_{args.net_type}_accuracy_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'))
    plt.show()

def plot_loss(losses, avg_losses, output_path, args):
    """
    Plot loss over epochs for each fold and the average loss.
    
    Parameters:
    - losses: List of lists containing loss values for each fold.
    - avg_losses: List of average loss values over epochs.
    - output_path: Directory path to save the plot.
    - args: Arguments containing model parameters for filename.
    """
    plt.figure(figsize=(10, 6))
    for i, loss in enumerate(losses):
        plt.plot(loss, label=f'Fold {i + 1}')
    plt.plot(avg_losses, color='blue', label='Average Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_path, f'loss_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'))
    plt.show()

def plot_precision_recall_curves(fold_results, save_path):
    plt.figure(figsize=(10, 8))

    all_recall_interp = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(all_recall_interp)

    for i, (true_labels, predicted_scores) in enumerate(fold_results):
        precision, recall, _ = precision_recall_curve(true_labels, predicted_scores)
        pr_auc = auc(recall, precision)
        
        # Interpolating precision at fixed recall values
        precision_interp = np.interp(all_recall_interp, recall[::-1], precision[::-1])
        mean_precision += precision_interp
        
        plt.plot(recall, precision, lw=2, alpha=0.3, label=f'PR fold {i + 1} (AUC = {pr_auc:.4f})')

    mean_precision /= len(fold_results)
    mean_pr_auc = auc(all_recall_interp, mean_precision)
    
    plt.plot(all_recall_interp, mean_precision, color='b', label=f'Mean PR (AUC = {mean_pr_auc:.4f})', lw=2, alpha=0.8)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    plt.close()
    
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    return roc_auc_score(labels, scores)

def compute_f1(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    pos_labels = np.ones(pos_score.shape[0])
    neg_labels = np.zeros(neg_score.shape[0])
    labels = np.concatenate([pos_labels, neg_labels])
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return f1_score(labels, preds_binary, zero_division=1) 

def compute_accuracy(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    pos_labels = np.ones(pos_score.shape[0])
    neg_labels = np.zeros(neg_score.shape[0])
    labels = np.concatenate([pos_labels, neg_labels])
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return accuracy_score(labels, preds_binary)

def compute_precision(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    pos_labels = np.ones(pos_score.shape[0])
    neg_labels = np.zeros(neg_score.shape[0])
    labels = np.concatenate([pos_labels, neg_labels])
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return precision_score(labels, preds_binary, zero_division=1) 

def compute_recall(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    pos_labels = np.ones(pos_score.shape[0])
    neg_labels = np.zeros(neg_score.shape[0])
    labels = np.concatenate([pos_labels, neg_labels])
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return recall_score(labels, preds_binary, zero_division=1) 

def compute_hits_k(pos_score, neg_score, k=10):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().cpu().numpy()
    ranked_scores = np.argsort(-scores)  # Rank in descending order
    top_k = ranked_scores[:k]
    return np.mean(labels[top_k])

def compute_map(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().cpu().numpy()
    ranked_indices = np.argsort(-scores)  # Rank in descending order
    sorted_labels = labels[ranked_indices]

    precisions = []
    relevant_docs = 0
    for i, label in enumerate(sorted_labels):
        if label == 1:
            relevant_docs += 1
            precisions.append(relevant_docs / (i + 1))
    
    if len(precisions) == 0:
        return 0.0
    
    return np.mean(precisions)

def compute_map_k(pos_score, neg_score, k=None):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().cpu().numpy()
    ranked_indices = np.argsort(-scores)  # Rank in descending order
    sorted_labels = labels[ranked_indices]

    if k is not None:
        sorted_labels = sorted_labels[:k]

    precisions = []
    relevant_docs = 0
    for i, label in enumerate(sorted_labels):
        if label == 1:
            relevant_docs += 1
            precisions.append(relevant_docs / (i + 1))

    if len(precisions) == 0:
        return 0.0

    return np.mean(precisions)

# Define new metric functions with confidence intervals

def compute_accuracy_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_accuracy(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        pos_labels = np.ones(pos_score.shape[0])
        neg_labels = np.zeros(neg_score.shape[0])
        labels = np.concatenate([pos_labels, neg_labels])
        threshold = 0.5
        preds_binary = (scores > threshold).astype(int)
        return accuracy_score(labels, preds_binary)
    
    initial_accuracy = compute_accuracy(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_accuracy, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_accuracy, error_range

def compute_precision_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_precision(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        pos_labels = np.ones(pos_score.shape[0])
        neg_labels = np.zeros(neg_score.shape[0])
        labels = np.concatenate([pos_labels, neg_labels])
        threshold = 0.5
        preds_binary = (scores > threshold).astype(int)
        return precision_score(labels, preds_binary, zero_division=1)
    
    initial_precision = compute_precision(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_precision, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_precision, error_range

def compute_f1_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_f1(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        pos_labels = np.ones(pos_score.shape[0])
        neg_labels = np.zeros(neg_score.shape[0])
        labels = np.concatenate([pos_labels, neg_labels])
        threshold = 0.5
        preds_binary = (scores > threshold).astype(int)
        return f1_score(labels, preds_binary, zero_division=1)
    
    initial_f1 = compute_f1(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_f1, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_f1, error_range

def compute_focalloss(pos_score, neg_score, alpha=1, gamma=2, device='cpu'):
    # Ensure both pos_score and neg_score are on the same device
    pos_score = pos_score.to(device)
    neg_score = neg_score.to(device)

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)

    BCE_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean().item()


def compute_focalloss_with_symmetrical_confidence(pos_score, neg_score, alpha=1, gamma=2, n_bootstraps=1000, confidence_level=0.95):
    initial_focal_loss = compute_focalloss(pos_score, neg_score, alpha, gamma)
    error_range = bootstrap_confidence_interval(
        lambda pos, neg: compute_focalloss(pos, neg, alpha, gamma),
        pos_score, neg_score, n_bootstraps, confidence_level
    )
    return initial_focal_loss, error_range

def compute_loss_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_loss(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        pos_labels = np.ones(pos_score.shape[0])
        neg_labels = np.zeros(neg_score.shape[0])
        labels = np.concatenate([pos_labels, neg_labels])
        threshold = 0.5
        preds_binary = (scores > threshold).astype(int)
        return loss_score(labels, preds_binary)
    
    initial_f1 = compute_loss(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_loss, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_f1, error_range

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_auc(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
        return roc_auc_score(labels, scores)
    
    initial_auc = compute_auc(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_auc, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_auc, error_range

def compute_recall_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_recall(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        pos_labels = np.ones(pos_score.shape[0])
        neg_labels = np.zeros(neg_score.shape[0])
        labels = np.concatenate([pos_labels, neg_labels])
        threshold = 0.5
        preds_binary = (scores > threshold).astype(int)
        return recall_score(labels, preds_binary, zero_division=1)
    
    initial_recall = compute_recall(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_recall, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_recall, error_range

def compute_map_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_map(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().cpu().numpy()
        ranked_indices = np.argsort(-scores)
        sorted_labels = labels[ranked_indices]

        precisions = []
        relevant_docs = 0
        for i, label in enumerate(sorted_labels):
            if label == 1:
                relevant_docs += 1
                precisions.append(relevant_docs / (i + 1))

        if len(precisions) == 0:
            return 0.0

        return np.mean(precisions)
    
    initial_map = compute_map(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_map, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_map, error_range

# Helper function to perform bootstrap resampling and calculate error range
def bootstrap_confidence_interval(metric_func, pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    metric_scores = []
    for _ in range(n_bootstraps):
        pos_sampled = resample(pos_score.cpu().numpy())
        neg_sampled = resample(neg_score.cpu().numpy())
        metric_scores.append(metric_func(torch.tensor(pos_sampled), torch.tensor(neg_sampled)))
    
    lower_bound = np.percentile(metric_scores, ((1 - confidence_level) / 2) * 100)
    upper_bound = np.percentile(metric_scores, (confidence_level + (1 - confidence_level) / 2) * 100)
    error_range = (upper_bound - lower_bound) / 2
    
    return error_range

def plot_scores(epochs, train_f1_scores, val_f1_scores, train_focal_loss_scores, val_focal_loss_scores, train_auc_scores, val_auc_scores, 
                train_map_scores, val_map_scores, train_recall_scores, val_recall_scores,
                train_acc_scores, val_acc_scores, train_precision_scores, val_precision_scores,
                output_path, args):

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_f1_scores, label='Training F1 Score')
    plt.plot(epochs, val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Scores over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'{args.model_type}_{args.net_type}_f1_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))

    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_focal_loss_scores, label='Training FocalLoss Score')
    plt.plot(epochs, val_focal_loss_scores, label='Validation FocalLoss Score')
    plt.xlabel('Epochs')
    plt.ylabel('FocalLoss Score')
    plt.title('Training and Validation FocalLoss Scores over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'{args.model_type}_{args.net_type}_loss_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))
    
    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_auc_scores, label='Training AUC')
    plt.plot(epochs, val_auc_scores, label='Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'{args.model_type}_{args.net_type}_auc_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))

    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_map_scores, label='Training mAP')
    plt.plot(epochs, val_map_scores, label='Validation mAP')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Training and Validation mAP over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'{args.model_type}_{args.net_type}_mAP_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))


    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_recall_scores, label='Training Recall')
    plt.plot(epochs, val_recall_scores, label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'{args.model_type}_{args.net_type}_recall_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))


    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_acc_scores, label='Training Accuracy')
    plt.plot(epochs, val_acc_scores, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'{args.model_type}_{args.net_type}_acc_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))
    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_precision_scores, label='Training Precision')
    plt.plot(epochs, val_precision_scores, label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'{args.model_type}_{args.net_type}_precision_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))

    plt.show()

