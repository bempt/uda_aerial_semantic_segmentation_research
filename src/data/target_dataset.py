import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class TargetDataset(Dataset):
    """Dataset class for target domain images (no labels)."""
    
    def __init__(self, images_dir, transform=None, target_size=(256, 256)):
        """
        Args:
            images_dir (str): Path to directory containing target domain images
            transform (callable, optional): Optional transform to be applied
            target_size (tuple): Target size for resizing (height, width)
        """
        self.images_dir = images_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get sorted list of image files
        self.images = sorted([
            f for f in os.listdir(images_dir) 
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        print(f"Found {len(self.images)} target domain images")
        if len(self.images) > 0:
            print(f"First target image: {self.images[0]}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get image for given index."""
        # Load image
        image_path = os.path.join(self.images_dir, self.images[idx])
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to target size
        if self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        return image