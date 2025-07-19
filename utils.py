import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across torch, numpy and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return torch device: GPU if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(model: torch.nn.Module, path: str):
    """Save model state dict to given path, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device=None):
    """Load state dict into model from path, map to device."""
    if device is None:
        device = get_device()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_metrics(true_labels, pred_labels, average='binary'):
    """Compute accuracy, precision, recall, f1-score."""
    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=average
    )
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_confusion_matrix(cm, labels, save_path=None, figsize=(6,6)):
    """
    Plot and optionally save a confusion matrix.
    Dependencies: seaborn, matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Please install seaborn and matplotlib to plot the confusion matrix.")
        return
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
