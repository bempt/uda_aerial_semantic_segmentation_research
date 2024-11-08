import os
import kaggle
from pathlib import Path

def download_semantic_drone_dataset(output_dir: str = "data/raw/semantic_drone"):
    """
    Downloads the Semantic Drone Dataset from Kaggle.
    
    Args:
        output_dir (str): Directory where the dataset will be downloaded
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download dataset using Kaggle API
    dataset_name = "bulentsiyah/semantic-drone-dataset"
    
    print(f"Downloading Semantic Drone Dataset to {output_dir}...")
    kaggle.api.dataset_download_files(
        dataset_name,
        path=output_dir,
        unzip=True
    )
    print("Download complete!")

if __name__ == "__main__":
    download_semantic_drone_dataset() 