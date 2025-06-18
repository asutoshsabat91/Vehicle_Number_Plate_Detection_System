import os
import kaggle
import shutil
from pathlib import Path

def download_dataset():
    """
    Download the Vehicle Number Plate Detection dataset from Kaggle.
    """
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Download dataset using kaggle API
    kaggle.api.dataset_download_files(
        'andrewmvd/car-plate-detection',
        path=data_dir,
        unzip=True
    )
    
    print(f"Dataset downloaded to: {data_dir}")
    
    # Organize the dataset
    organize_dataset(data_dir)

def organize_dataset(data_dir: Path):
    """
    Organize the downloaded dataset into a proper structure.
    """
    # Create directories for training and validation
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Get all image files from the dataset
    image_files = []
    
    # Look for images in common directories
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(data_dir.rglob(ext))
    
    if not image_files:
        print("No image files found in the dataset!")
        return
    
    # Split into train and validation sets (80-20 split)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Move files to their respective directories
    for file in train_files:
        shutil.copy2(str(file), str(train_dir / file.name))
    for file in val_files:
        shutil.copy2(str(file), str(val_dir / file.name))
    
    print(f"Organized {len(train_files)} training images and {len(val_files)} validation images")

if __name__ == '__main__':
    download_dataset() 