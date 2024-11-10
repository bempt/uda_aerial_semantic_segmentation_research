"""Dataset class for semantic drone dataset"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class DroneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Dataset for semantic segmentation.
        
        Args:
            images_dir (str): Directory with input images
            masks_dir (str): Directory with segmentation masks
            transform (callable, optional): Optional transform to be applied
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
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