import torch
import numpy as np

class DomainAdaptationMetrics:
    """
    Metrics specific to domain adaptation performance tracking.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.source_domain_accuracy = 0
        self.target_domain_accuracy = 0
        self.domain_confusion_score = 0
        self.num_batches = 0
        
    def update(self, source_pred, target_pred):
        """
        Update metrics with new predictions.
        
        Args:
            source_pred (torch.Tensor): Discriminator predictions for source domain
            target_pred (torch.Tensor): Discriminator predictions for target domain
        """
        # Convert predictions to binary (threshold at 0.5)
        source_binary = (source_pred > 0.5).float()
        target_binary = (target_pred > 0.5).float()
        
        # Source domain accuracy (should be classified as 0)
        self.source_domain_accuracy += (source_binary == 0).float().mean().item()
        
        # Target domain accuracy (should be classified as 1)
        self.target_domain_accuracy += (target_binary == 1).float().mean().item()
        
        # Domain confusion score (closer to 0.5 means better confusion)
        source_confusion = abs(source_pred.mean().item() - 0.5)
        target_confusion = abs(target_pred.mean().item() - 0.5)
        self.domain_confusion_score += 1.0 - (source_confusion + target_confusion) / 2
        
        self.num_batches += 1
        
    def get_metrics(self):
        """
        Get current metric values.
        
        Returns:
            dict: Dictionary of metric names and values
        """
        if self.num_batches == 0:
            return {
                'source_domain_acc': 0.0,
                'target_domain_acc': 0.0,
                'domain_confusion': 0.0
            }
            
        return {
            'source_domain_acc': f'{self.source_domain_accuracy / self.num_batches:.4f}',
            'target_domain_acc': f'{self.target_domain_accuracy / self.num_batches:.4f}',
            'domain_confusion': f'{self.domain_confusion_score / self.num_batches:.4f}'
        } 