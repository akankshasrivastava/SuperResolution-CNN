import unittest
import torch
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Now import from src.data instead of just data
from src.data.dataset import SuperResolutionDataset
from tests.test_base import BaseTestCase


class TestSuperResolutionDataset(BaseTestCase):
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()  # Call parent's setUp

        # Create dummy images
        self.hr_dir = self.test_dir / 'test_hr'
        self.lr_dir = self.test_dir / 'test_lr'
        self.hr_dir.mkdir(exist_ok=True)
        self.lr_dir.mkdir(exist_ok=True)

        # Create some test images
        self.num_test_images = 3
        for i in range(self.num_test_images):
            # Create random BGR images
            hr_img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
            lr_img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)

            # Save images
            cv2.imwrite(str(self.hr_dir / f'img_{i}.png'), hr_img)
            cv2.imwrite(str(self.lr_dir / f'img_{i}.png'), lr_img)

        # Create dataset
        self.dataset = SuperResolutionDataset(self.hr_dir, self.lr_dir)

    def test_dataset_length(self):
        """Test if dataset reports correct length"""
        self.assertEqual(len(self.dataset), self.num_test_images)

    def test_dataset_getitem(self):
        """Test if dataset returns correct tensor types and shapes"""
        lr_img, hr_img = self.dataset[0]

        # Check types
        self.assertIsInstance(lr_img, torch.FloatTensor)
        self.assertIsInstance(hr_img, torch.FloatTensor)

        # Check shapes
        self.assertEqual(lr_img.dim(), 3)  # C, H, W
        self.assertEqual(hr_img.dim(), 3)

        # Check value range (normalized to [-1, 1])
        self.assertTensorInRange(lr_img, -1, 1)
        self.assertTensorInRange(hr_img, -1, 1)

    def test_dataset_compatibility(self):
        """Test if dataset can be used with DataLoader"""
        from torch.utils.data import DataLoader
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        self.assertEqual(len(batch), 2)  # lr and hr


if __name__ == '__main__':
    unittest.main()