import torch
import numpy as np
import torch.nn.functional as F

def create_overlay(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Create an overlay of an image and a segmentation mask.
    
    Args:
        image: Input image tensor (C, H, W)
        mask: Segmentation mask tensor (H, W)
        alpha: Transparency of the overlay (0-1)
        
    Returns:
        torch.Tensor: Overlay image tensor (C, H, W)
    """
    # Ensure inputs are on CPU and in correct format
    image = image.cpu()
    mask = mask.cpu()
    
    # Create colored mask
    num_classes = int(mask.max().item()) + 1
    colored_mask = torch.zeros_like(image)
    
    # Create different colors for each class
    colors = torch.tensor([
        [0.0, 0.0, 0.0],  # Background - black
        [1.0, 0.0, 0.0],  # Class 1 - red
        [0.0, 1.0, 0.0],  # Class 2 - green
        [0.0, 0.0, 1.0],  # Class 3 - blue
        [1.0, 1.0, 0.0],  # Class 4 - yellow
        [1.0, 0.0, 1.0],  # Class 5 - magenta
        [0.0, 1.0, 1.0],  # Class 6 - cyan
        [0.5, 0.5, 0.5],  # Class 7 - gray
    ])
    
    # Ensure we have enough colors
    while len(colors) < num_classes:
        colors = torch.cat([colors, torch.rand(8, 3)])
    
    # Create one-hot encoded mask
    mask_one_hot = F.one_hot(mask.long(), num_classes).permute(2, 0, 1)
    
    # Apply colors to mask
    for class_idx in range(num_classes):
        for channel in range(3):
            colored_mask[channel] += mask_one_hot[class_idx] * colors[class_idx][channel]
    
    # Create overlay
    overlay = image * (1 - alpha) + colored_mask * alpha
    overlay = torch.clamp(overlay, 0, 1)
    
    return overlay 