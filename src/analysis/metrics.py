import torch
import numpy as np
from typing import Optional, Union, List

class SegmentationMetrics:
    """Compute semantic segmentation metrics including IoU, pixel accuracy, and F1-score."""
    
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        """
        Args:
            num_classes: Number of classes in the segmentation task
            ignore_index: Optional index to ignore (e.g., background)
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def _fast_hist(self, pred: torch.Tensor, true: torch.Tensor) -> np.ndarray:
        """Compute confusion matrix."""
        mask = (true >= 0) & (true < self.num_classes)
        if self.ignore_index is not None:
            mask = mask & (true != self.ignore_index)
            
        hist = torch.bincount(
            self.num_classes * true[mask].long() + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist.cpu().numpy()
    
    def batch_iou(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute IoU for each class and mean IoU."""
        hist = self._fast_hist(predictions.flatten(), targets.flatten())
        
        # Compute IoU for each class
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-7)
        mean_iou = np.nanmean(iu)
        
        # Create results dictionary
        results = {
            'mean_iou': mean_iou,
            'class_iou': {i: iou for i, iou in enumerate(iu)}
        }
        return results
    
    def pixel_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute pixel-wise accuracy."""
        mask = targets != self.ignore_index if self.ignore_index is not None else torch.ones_like(targets, dtype=torch.bool)
        correct = torch.sum((predictions == targets) & mask).item()
        total = torch.sum(mask).item()
        return correct / (total + 1e-7)
    
    def f1_score(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 class_index: Optional[int] = None) -> Union[float, List[float]]:
        """Compute F1 score for specified class or all classes."""
        hist = self._fast_hist(predictions.flatten(), targets.flatten())
        
        if class_index is not None:
            tp = hist[class_index, class_index]
            fp = hist[:, class_index].sum() - tp
            fn = hist[class_index, :].sum() - tp
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
            return f1
        
        # Compute F1 for all classes
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        f1_scores = 2 * tp / (2 * tp + fp + fn + 1e-7)
        return f1_scores.tolist() 