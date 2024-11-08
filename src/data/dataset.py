"""Dataset class for semantic drone dataset"""

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

class DroneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir (str): Path to directory with original images
            masks_dir (str): Path to directory with semantic mask images
            transform: Optional transform to be applied to image/mask pairs
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Get sorted filenames
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
        
        # Print debug info
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
        # Load image and mask
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        # Read image (BGR)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask - keep as is for multi-class segmentation
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        
        # Apply transforms if any
        if self.transform is not None:
            # Keep mask as numpy array for albumentations
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']  # This will be a torch tensor
            mask = transformed['mask']    # This will be a numpy array
        
        # Convert mask to tensor after transforms
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        
        # Ensure mask is long type for criterion
        mask = mask.long()
        
        return image, mask