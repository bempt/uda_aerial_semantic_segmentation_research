import shutil
from pathlib import Path
import random
import numpy as np
from typing import Union, List
import logging

def create_sample_dataset(
    source_dir: Union[str, Path] = "data/raw/semantic_drone",
    output_dir: Union[str, Path] = "data/sample/semantic_drone",
    num_samples: int = 10,
    seed: int = 42
) -> None:
    """
    Creates a smaller sample dataset from the Semantic Drone Dataset.
    
    Args:
        source_dir: Directory containing the original dataset
        output_dir: Directory where the sample dataset will be saved
        num_samples: Number of images to include in the sample dataset
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Define source and target directories based on actual dataset structure
    original_dirs = {
        'images': source_dir / 'dataset' / 'semantic_drone_dataset' / 'original_images',
        'labels': source_dir / 'dataset' / 'semantic_drone_dataset' / 'label_images_semantic'
    }
    
    sample_dirs = {
        'images': output_dir / 'original_images',
        'labels': output_dir / 'label_images_semantic'
    }
    
    # Create output directories
    for dir_path in sample_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of all image files
    image_files = list(original_dirs['images'].glob('*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No images found in {original_dirs['images']}")
    
    # Randomly select samples
    num_samples = min(num_samples, len(image_files))
    selected_files = random.sample(image_files, num_samples)
    
    # Copy selected files
    for img_path in selected_files:
        # Get corresponding label file
        label_path = original_dirs['labels'] / img_path.name.replace('.jpg', '.png')
        
        if not label_path.exists():
            logging.warning(f"Label not found for {img_path.name}, skipping...")
            continue
        
        # Copy image and label
        shutil.copy2(img_path, sample_dirs['images'] / img_path.name)
        shutil.copy2(label_path, sample_dirs['labels'] / label_path.name)
        
    print(f"Created sample dataset with {num_samples} images in {output_dir}")
    
    # Copy class mapping file if it exists
    class_mapping = source_dir / 'class_dict_seg.csv'  # Updated filename based on repo structure
    if class_mapping.exists():
        shutil.copy2(class_mapping, output_dir / 'class_dict_seg.csv')

if __name__ == "__main__":
    create_sample_dataset()