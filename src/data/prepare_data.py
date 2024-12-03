import os
import shutil
import re
import cv2
from tqdm import tqdm
from pathlib import Path

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def create_directory(path):
    os.makedirs(path, exist_ok=True)
    return path

def reorganize_dataset():
    print("Starting dataset reorganization...")
    
    # Define paths
    base_path = Path('data')
    raw_path = base_path / 'Raw Data'
    
    # Create new directory structure
    splits = ['train', 'val', 'test']
    resolutions = ['hr', 'lr']
    
    for split in splits:
        for res in resolutions:
            create_directory(base_path / split / res)
    
    # Get all images from Raw Data
    files = sorted_alphanumeric([f for f in os.listdir(raw_path / 'high_res') if f.endswith('.png')])
    
    # Define split points
    train_end = 700
    val_end = 830  # 700 + 130
    
    print("Processing images...")
    for idx, filename in enumerate(tqdm(files)):
        if idx >= 855:  # Stop at 855 as in the paper
            break
            
        # Load images
        hr_img = cv2.imread(str(raw_path / 'high_res' / filename))
        lr_img = cv2.imread(str(raw_path / 'low_res' / filename))
        
        if hr_img is None or lr_img is None:
            print(f"Warning: Could not read image {filename}")
            continue
        
        # Determine which split to use
        if idx < train_end:
            split = 'train'
        elif idx < val_end:
            split = 'val'
        else:
            split = 'test'
        
        # Save with new names
        new_filename = f"{idx:04d}.png"
        cv2.imwrite(str(base_path / split / 'hr' / new_filename), hr_img)
        cv2.imwrite(str(base_path / split / 'lr' / new_filename), lr_img)
    
    # Remove old directories
    for dir_name in ['high_res', 'low_res']:
        if (base_path / 'train' / dir_name).exists():
            shutil.rmtree(base_path / 'train' / dir_name)
        if (base_path / 'val' / dir_name).exists():
            shutil.rmtree(base_path / 'val' / dir_name)
    
    # Print statistics
    print("\nDataset reorganization completed!")
    for split in splits:
        hr_count = len(list((base_path / split / 'hr').glob('*.png')))
        lr_count = len(list((base_path / split / 'lr').glob('*.png')))
        print(f"{split} set: {hr_count} hr images, {lr_count} lr images")

if __name__ == "__main__":
    reorganize_dataset()
