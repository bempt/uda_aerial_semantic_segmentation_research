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
            targets: Ground truth in one-hot format (B, C, H, W)
        Returns:
            Dice loss value
        """
        batch_size = predictions.size(0)
        num_classes = predictions.size(1)
        
        # Apply softmax to predictions
        pred_probs = F.softmax(predictions, dim=1)
        
        # Reshape tensors to (B, C, H*W)
        pred_flat = pred_probs.reshape(batch_size, num_classes, -1)
        targets_flat = targets.reshape(batch_size, num_classes, -1)
        
        # Calculate intersection and union per class and batch
        intersection = (pred_flat * targets_flat).sum(2)  # Sum over H*W
        union = pred_flat.sum(2) + targets_flat.sum(2)   # Sum over H*W
        
        # Calculate Dice coefficient per class
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes and batch
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