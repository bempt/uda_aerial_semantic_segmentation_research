import os
import shutil
from pathlib import Path

def create_sample_holyrood():
    """
    Creates a sample Holyrood dataset by copying a few representative images 
    from the target dataset to the samples directory.
    """
    # Define source and destination paths
    source_dir = Path("data/target/holyrood")
    dest_dir = Path("data/sample/holyrood")
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Select representative images (different angles, lighting conditions)
    sample_images = [
        "DJI_0189.JPG",  # Front view
        "DJI_0306.JPG",  # Side view
        "DJI_0419.JPG",  # Aerial view
        "DJI_0514.JPG",  # Different lighting
        "DJI_0562.JPG",  # Different angle
        "DJI_0646.JPG",  # Different perspective
        "DJI_0689.JPG",  # Different time of day
        "DJI_0722.JPG",  # Different weather condition
    ]
    
    # Copy selected images
    for image_name in sample_images:
        source_file = source_dir / image_name
        dest_file = dest_dir / image_name
        
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            print(f"Copied {image_name}")
        else:
            print(f"Warning: Source file {image_name} not found")
    
    # Count files in destination directory
    num_files = len(list(dest_dir.glob("*.JPG")))
    print(f"\nCreated sample dataset with {num_files} images in {dest_dir}")
    
    return num_files

if __name__ == "__main__":
    create_sample_holyrood() 