import numpy as np
import torch
import torch.nn.functional as F

class DomainAdaptationMetrics:
    """Track metrics for domain adaptation training."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.source_correct = 0
        self.source_total = 0
        self.target_correct = 0
        self.target_total = 0
        self.domain_entropy_sum = 0.0
        self.feature_alignment_sum = 0.0
        self.n_batches = 0
        
    def update(self, source_pred, target_pred, source_features=None, target_features=None):
        """Update all metrics in one call."""
        self.update_domain_accuracy(source_pred, target_pred)
        self.update_confusion_metrics(source_features, target_features, 
                                    torch.cat([source_pred, target_pred], dim=0))
        
    def update_domain_accuracy(self, source_pred, target_pred):
        """Update domain classification accuracy metrics."""
        # Source domain accuracy (should be classified as 1)
        self.source_correct += (source_pred >= 0.5).sum().item()
        self.source_total += source_pred.size(0)
        
        # Target domain accuracy (should be classified as 0)
        self.target_correct += (target_pred < 0.5).sum().item()
        self.target_total += target_pred.size(0)
        
    def update_confusion_metrics(self, source_features, target_features, domain_predictions):
        """Update domain confusion metrics."""
        # Calculate domain entropy
        domain_probs = torch.sigmoid(domain_predictions)
        entropy = -domain_probs * torch.log(domain_probs + 1e-10) - \
                 (1 - domain_probs) * torch.log(1 - domain_probs + 1e-10)
        self.domain_entropy_sum += entropy.mean().item()
        
        # Calculate feature alignment (cosine similarity between source and target features)
        if source_features is not None and target_features is not None:
            source_norm = F.normalize(source_features.mean(0), dim=0)
            target_norm = F.normalize(target_features.mean(0), dim=0)
            alignment = F.cosine_similarity(source_norm, target_norm, dim=0)
            self.feature_alignment_sum += alignment.item()
        
        self.n_batches += 1
        
    def get_metrics(self):
        """Get current metrics as dictionary."""
        source_acc = self.source_correct / max(self.source_total, 1)
        target_acc = self.target_correct / max(self.target_total, 1)
        domain_confusion = self.domain_entropy_sum / max(self.n_batches, 1)
        
        return {
            'source_domain_acc': f'{source_acc:.4f}',
            'target_domain_acc': f'{target_acc:.4f}',
            'domain_confusion': f'{domain_confusion:.4f}'
        }
        
    def get_confusion_metrics(self):
        """Get detailed confusion metrics."""
        avg_entropy = self.domain_entropy_sum / max(self.n_batches, 1)
        avg_alignment = self.feature_alignment_sum / max(self.n_batches, 1)
        
        return {
            'domain_entropy': avg_entropy,
            'feature_alignment': avg_alignment
        }