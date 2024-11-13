import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np

class AdversarialLoss:
    def __init__(self, lambda_adv=0.001):
        """
        Adversarial loss for domain adaptation.
        
        Args:
            lambda_adv (float): Weight for adversarial loss (default: 0.001)
        """
        self.lambda_adv = lambda_adv
        self.bce_loss = nn.BCELoss()
        
    def discriminator_loss(self, source_pred, target_pred):
        """
        Calculate discriminator loss.
        
        Args:
            source_pred (torch.Tensor): Discriminator predictions for source domain
            target_pred (torch.Tensor): Discriminator predictions for target domain
            
        Returns:
            torch.Tensor: Discriminator loss
        """
        device = source_pred.device
        source_label = torch.ones_like(source_pred).to(device)
        target_label = torch.zeros_like(target_pred).to(device)
        
        source_loss = self.bce_loss(source_pred, source_label)
        target_loss = self.bce_loss(target_pred, target_label)
        
        return (source_loss + target_loss) / 2
        
    def generator_loss(self, target_pred):
        """
        Calculate generator (segmentation model) adversarial loss.
        The generator tries to fool the discriminator by making target predictions close to 0.
        
        Args:
            target_pred (torch.Tensor): Discriminator predictions for target domain
            
        Returns:
            torch.Tensor: Generator adversarial loss
        """
        device = target_pred.device
        target_label = torch.ones_like(target_pred).to(device)
        return self.lambda_adv * self.bce_loss(target_pred, target_label)

class ConsistencyLoss(nn.Module):
    """
    Consistency loss for unsupervised learning.
    Enforces consistent predictions between different augmented versions of the same image.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred1, pred2):
        """
        Calculate consistency loss between two predictions.
        
        Args:
            pred1: First prediction tensor (B, C, H, W)
            pred2: Second prediction tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Consistency loss value
        """
        # Apply softmax with temperature
        prob1 = F.softmax(pred1 / self.temperature, dim=1)
        prob2 = F.softmax(pred2 / self.temperature, dim=1)
        
        # Calculate KL divergence in both directions
        loss_kl_1 = F.kl_div(
            F.log_softmax(pred1 / self.temperature, dim=1),
            prob2,
            reduction='batchmean'
        )
        loss_kl_2 = F.kl_div(
            F.log_softmax(pred2 / self.temperature, dim=1),
            prob1,
            reduction='batchmean'
        )
        
        # Take the mean of both directions
        return (loss_kl_1 + loss_kl_2) / 2
    
    def get_similarity_matrix(self, pred1, pred2):
        """
        Calculate similarity matrix between predictions for visualization.
        
        Args:
            pred1: First prediction tensor (B, C, H, W)
            pred2: Second prediction tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Similarity matrix (B, H, W)
        """
        prob1 = F.softmax(pred1, dim=1)
        prob2 = F.softmax(pred2, dim=1)
        
        # Calculate cosine similarity across channel dimension
        similarity = F.cosine_similarity(prob1, prob2, dim=1)
        return similarity

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation with proper tensor handling
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        """
        Calculate Dice Loss
        Args:
            predictions: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W) or one-hot (B, C, H, W)
        Returns:
            Dice loss value
        """
        batch_size = predictions.size(0)
        num_classes = predictions.size(1)
        
        # Apply softmax to predictions
        pred_probs = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot if needed
        if targets.dim() == 3:
            # Reshape targets to (B, H*W) for one_hot encoding
            targets_flat = targets.view(batch_size, -1)
            # Convert to one-hot
            targets = F.one_hot(targets_flat.long(), num_classes)
            # Reshape back to (B, C, H, W)
            h, w = predictions.shape[2:]
            targets = targets.view(batch_size, h, w, num_classes)
            targets = targets.permute(0, 3, 1, 2).float()
        
        # Calculate intersection and union
        intersection = (pred_probs * targets).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return mean loss
        return 1.0 - dice.mean()

class WeightedSegmentationLoss(nn.Module):
    """
    Weighted segmentation loss that handles class imbalance
    """
    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        alpha: float = 0.25,  # For focal loss component
        gamma: float = 2.0,   # For focal loss component
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('class_weights', 
                           class_weights if class_weights is not None 
                           else torch.ones(num_classes))
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.dice_loss = DiceLoss()
        
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss component
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
        
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        domain_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Compute weighted loss combining focal and dice loss
        
        Args:
            inputs: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            domain_weight: Weight for this domain's samples
            
        Returns:
            Combined weighted loss
        """
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute losses
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets_one_hot)
        
        # Combine losses with domain weight
        return domain_weight * (focal + dice)

def calculate_class_weights(
    dataset,
    num_classes: int,
    method: str = 'effective_samples'
) -> torch.Tensor:
    """
    Calculate class weights based on class frequencies
    
    Args:
        dataset: Dataset to analyze
        num_classes: Number of classes
        method: Weighting method ('effective_samples' or 'inverse_freq')
        
    Returns:
        Tensor of class weights
    """
    # Count pixels per class
    class_counts = torch.zeros(num_classes)
    for _, mask in dataset:
        for class_idx in range(num_classes):
            class_counts[class_idx] += (mask == class_idx).sum().item()
            
    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)
    
    if method == 'effective_samples':
        # Effective number of samples weighting
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
    else:
        # Inverse frequency weighting
        weights = 1.0 / class_counts
        
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return weights

class FineTuningLoss(nn.Module):
    """
    Combined loss for Phase 3 unsupervised fine-tuning.
    Combines consistency loss, domain confusion loss, and optional supervised loss.
    """
    def __init__(
        self,
        consistency_weight: float = 1.0,
        domain_weight: float = 0.1,
        supervised_weight: float = 0.1,
        rampup_length: int = 40,
        temperature: float = 0.5
    ):
        super().__init__()
        self.consistency_loss = ConsistencyLoss(temperature=temperature)
        self.domain_loss = AdversarialLoss(lambda_adv=domain_weight)
        self.supervised_loss = DiceLoss()
        
        self.consistency_weight = consistency_weight
        self.domain_weight = domain_weight
        self.supervised_weight = supervised_weight
        self.rampup_length = rampup_length
        
    def rampup(self, epoch: int) -> float:
        """Calculate rampup weight for loss components"""
        if epoch >= self.rampup_length:
            return 1.0
        else:
            # Smooth rampup from 0 to 1
            return float(epoch) / self.rampup_length
            
    def forward(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
        domain_pred: torch.Tensor,
        epoch: int,
        supervised_pred: Optional[torch.Tensor] = None,
        supervised_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined fine-tuning loss.
        
        Args:
            pred1: First prediction from augmented image (B, C, H, W)
            pred2: Second prediction from differently augmented image (B, C, H, W)
            domain_pred: Domain classifier prediction (B, 1)
            epoch: Current epoch for loss scheduling
            supervised_pred: Optional supervised prediction (B, C, H, W)
            supervised_target: Optional ground truth (B, H, W)
            
        Returns:
            Dictionary containing total loss and individual components
        """
        # Calculate rampup weight
        rampup_weight = self.rampup(epoch)
        
        # Consistency loss between predictions
        consistency = self.consistency_loss(pred1, pred2)
        weighted_consistency = consistency * self.consistency_weight * rampup_weight
        
        # Domain confusion loss
        domain_confusion = self.domain_loss.generator_loss(domain_pred)
        weighted_domain = domain_confusion * self.domain_weight * rampup_weight
        
        # Initialize total loss
        total_loss = weighted_consistency + weighted_domain
        
        # Initialize supervised loss component
        supervised = torch.tensor(0.0, device=pred1.device)
        
        # Add supervised loss if available
        if supervised_pred is not None and supervised_target is not None:
            # Ensure supervised_target is long type for cross entropy
            if supervised_target.dtype != torch.long:
                supervised_target = supervised_target.long()
            supervised = self.supervised_loss(supervised_pred, supervised_target)
            weighted_supervised = supervised * self.supervised_weight
            total_loss = total_loss + weighted_supervised
        
        return {
            'total': total_loss,
            'consistency': consistency.detach(),
            'domain_confusion': domain_confusion.detach(),
            'supervised': supervised.detach(),
            'rampup_weight': torch.tensor(rampup_weight)
        }