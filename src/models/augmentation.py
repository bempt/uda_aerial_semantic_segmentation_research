"""Data augmentation functions"""

import torch
from torchvision import transforms

def get_training_augmentation():
    """
    Get training augmentations using torchvision transforms.
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_validation_augmentation():
    """
    Get validation augmentations using torchvision transforms.
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]) 