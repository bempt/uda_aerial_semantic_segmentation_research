"""Dataset class for semantic drone dataset"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.models.config import Config

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
        
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {images_dir}")
            
        print(f"Found {len(self.image_files)} images and {len(self.image_files)} masks")
        print(f"First image: {self.image_files[0]}")
        print(f"First mask: {self.image_files[0].replace('.jpg', '.png')}")
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(
            self.masks_dir, 
            self.image_files[idx].replace('.jpg', '.png')
        )
        
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        
        # Load and convert mask
        mask = Image.open(mask_path).convert('L')  # Load as grayscale
        
        # Apply transforms to image
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default transform for image
            image = transforms.Compose([
                transforms.Resize(Config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=Config.NORMALIZE_MEAN,
                    std=Config.NORMALIZE_STD
                )
            ])(image)
        
        # Transform mask
        mask_transform = transforms.Compose([
            transforms.Resize(
                Config.IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])
        mask = mask_transform(mask)
        
        # Convert mask to proper shape (H, W) and type
        mask = mask.squeeze(0)  # Remove channel dimension
        mask = mask.long()  # Convert to long type for CrossEntropyLoss
        
        return image, mask