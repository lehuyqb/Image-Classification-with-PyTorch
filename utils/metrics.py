import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from .data_loader import CLASSES

class MetricsTracker:
    """
    Tracks and computes various metrics for model evaluation.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.running_loss = 0.0
        self.running_corrects = 0
        self.total_samples = 0
        self.all_preds = []
        self.all_labels = []
    
    def update(self, outputs, labels, loss):
        """
        Update metrics with batch results
        
        Args:
            outputs (torch.Tensor): Model outputs
            labels (torch.Tensor): True labels
            loss (float): Batch loss
        """
        preds = torch.argmax(outputs, dim=1)
        self.running_loss += loss.item() * labels.size(0)
        self.running_corrects += torch.sum(preds == labels).item()
        self.total_samples += labels.size(0)
        
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
    
    def get_metrics(self):
        """
        Compute final metrics
        
        Returns:
            dict: Dictionary containing various metrics
        """
        metrics = {
            'loss': self.running_loss / self.total_samples,
            'accuracy': self.running_corrects / self.total_samples
        }
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=CLASSES,
                   yticklabels=CLASSES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def print_classification_report(self):
        """Print detailed classification report"""
        print("\nClassification Report:")
        print(classification_report(
            self.all_labels,
            self.all_preds,
            target_names=CLASSES
        ))

def accuracy(output, target):
    """
    Compute accuracy for a single batch
    
    Args:
        output (torch.Tensor): Model output
        target (torch.Tensor): True labels
        
    Returns:
        float: Accuracy value
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0) 