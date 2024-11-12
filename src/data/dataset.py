"""Dataset class for semantic drone dataset"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import Dict
from tqdm.auto import tqdm

class DroneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, balance_classes=True):
        """
        Dataset for semantic segmentation with class balancing.
        
        Args:
            images_dir (str): Directory with input images
            masks_dir (str): Directory with segmentation masks
            transform (callable, optional): Optional transform to be applied
            balance_classes (bool): Whether to use class balancing
        """
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.balance_classes = balance_classes
        
        # Get sorted lists of files
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
        
        # Print some info
        print(f"Found {len(self.images)} images and {len(self.masks)} masks")
        if len(self.images) > 0:
            print(f"First image: {self.images[0]}")
            print(f"First mask: {self.masks[0]}")
        
        # Verify matching pairs
        assert len(self.images) == len(self.masks), \
            f"Number of images ({len(self.images)}) != number of masks ({len(self.masks)})"
        
        # Calculate class statistics if balancing enabled
        if balance_classes:
            print("Calculating class statistics...")
            self.class_stats = self._calculate_class_stats()
            self.sample_weights = self._calculate_sample_weights()
    
    def _calculate_class_stats(self) -> Dict[int, int]:
        """Calculate pixel counts per class"""
        class_counts = {}
        
        for mask_name in tqdm(self.masks, desc="Processing masks"):
            mask_path = os.path.join(self.masks_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            unique, counts = np.unique(mask, return_counts=True)
            for class_idx, count in zip(unique, counts):
                if class_idx not in class_counts:
                    class_counts[class_idx] = 0
                class_counts[class_idx] += count
                
        return class_counts
    
    def _calculate_sample_weights(self) -> np.ndarray:
        """Calculate sample weights for balanced sampling"""
        weights = np.zeros(len(self))
        
        for idx, mask_name in enumerate(self.masks):
            mask_path = os.path.join(self.masks_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Count pixels per class in this sample
            unique, counts = np.unique(mask, return_counts=True)
            
            # Calculate weight as inverse of class frequencies
            sample_weight = 0
            for class_idx, count in zip(unique, counts):
                class_freq = self.class_stats[class_idx] / sum(self.class_stats.values())
                sample_weight += (count / mask.size) * (1 / class_freq)
                
            weights[idx] = sample_weight
            
        return weights / weights.sum()
    
    def get_sampler(self, indices=None) -> WeightedRandomSampler:
        """
        Get weighted sampler for balanced sampling.
        
        Args:
            indices: Optional indices for subset sampling (for split datasets)
            
        Returns:
            WeightedRandomSampler instance or None
        """
        if not self.balance_classes:
            return None
            
        if indices is not None:
            # Use only the weights for the specified indices
            weights = self.sample_weights[indices]
        else:
            weights = self.sample_weights
            
        # Normalize weights
        weights = weights / weights.sum()
            
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        """Get image and mask for given index."""
        # Load image
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        # Read image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to long tensor if it's still a numpy array
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor) and mask.dtype != torch.long:
            mask = mask.long()
        
        return image, mask