import os
import zipfile
import shutil
from pathlib import Path

def prepare_holyrood_dataset():
    """
    Extract and organize the Holyrood dataset from zip files into the target directory.
    Only extracts if target directory is empty or doesn't exist.
    """
    # Define paths
    raw_dir = Path("data/raw/holyrood_october_2020")
    target_dir = Path("data/target/holyrood")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset is already prepared
    image_count = len(list(target_dir.glob("*.*")))
    if image_count > 0:
        print(f"Dataset already prepared with {image_count} images in {target_dir}")
        return image_count
    
    # List of zip files to process
    zip_files = [
        "MavicProCamData-20241109T091558Z-002.zip",
        "MavicProCamData-20241109T091558Z-003.zip",
        "MavicProCamData-20241109T092507Z-001.zip"
    ]
    
    # Create temporary extraction directory
    temp_dir = raw_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Extract all zip files
        for zip_file in zip_files:
            zip_path = raw_dir / zip_file
            if not zip_path.exists():
                raise FileNotFoundError(f"Zip file not found: {zip_path}")
            
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        
        # Move all image files to target directory
        print("Organizing files...")
        image_extensions = {'.jpg', '.jpeg', '.png'}
        
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    src_path = Path(root) / file
                    dst_path = target_dir / file
                    shutil.copy2(src_path, dst_path)
        
        print(f"Dataset prepared successfully in {target_dir}")
        
    finally:
        # Cleanup: remove temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
    # Verify dataset
    image_count = len(list(target_dir.glob("*.*")))
    if image_count == 0:
        raise RuntimeError("No images were extracted to target directory")
    
    return image_count

if __name__ == "__main__":
    count = prepare_holyrood_dataset()
    print(f"Total images prepared: {count}") 