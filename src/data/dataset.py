import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        super().__init__()
        self.hr_files = sorted(list(hr_dir.glob('*.png')))
        self.lr_dir = lr_dir
        
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # Load images
        hr_path = self.hr_files[idx]
        lr_path = self.lr_dir / hr_path.name
        
        # Load images
        hr_img = cv2.imread(str(hr_path))
        lr_img = cv2.imread(str(lr_path))
        
        # Convert to YCbCr
        hr_img_ycrcb = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        lr_img_ycrcb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2YCrCb)
        
        # Extract Y channel
        hr_y = hr_img_ycrcb[:, :, 0]
        lr_y = lr_img_ycrcb[:, :, 0]
        
        # Normalize to [-1, 1]
        hr_y = hr_y.astype(np.float32) / 127.5 - 1
        lr_y = lr_y.astype(np.float32) / 127.5 - 1
        
        # Add channel dimension
        hr_y = np.expand_dims(hr_y, axis=0)
        lr_y = np.expand_dims(lr_y, axis=0)
        
        return torch.FloatTensor(lr_y), torch.FloatTensor(hr_y)
