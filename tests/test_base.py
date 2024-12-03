import unittest
import torch
import numpy as np
from pathlib import Path


class BaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test cases"""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Create temporary directories for test data
        cls.test_dir = Path('tests/test_data')
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy data dimensions
        cls.batch_size = 2
        cls.channels = 3
        cls.height = 32
        cls.width = 32

        # Create dummy tensors
        cls.dummy_input = torch.randn(cls.batch_size, cls.channels, cls.height, cls.width)
        cls.dummy_target = torch.randn(cls.batch_size, cls.channels, cls.height, cls.width)

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        # Remove test directory and all contents
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def assertTensorShape(self, tensor, expected_shape, msg=None):
        """Assert that a tensor has the expected shape"""
        self.assertEqual(tuple(tensor.shape), expected_shape, msg)

    def assertTensorInRange(self, tensor, min_val, max_val, msg=None):
        """Assert that all values in a tensor are within the expected range"""
        self.assertTrue(torch.all((tensor >= min_val) & (tensor <= max_val)), msg)


class TestBaseSetup(BaseTestCase):
    """Test cases to verify the base testing infrastructure"""

    def test_directory_creation(self):
        """Test if test directory is created properly"""
        self.assertTrue(self.test_dir.exists())
        self.assertTrue(self.test_dir.is_dir())

    def test_dummy_tensors(self):
        """Test if dummy tensors are created with correct shapes"""
        expected_shape = (self.batch_size, self.channels, self.height, self.width)
        self.assertTensorShape(self.dummy_input, expected_shape)
        self.assertTensorShape(self.dummy_target, expected_shape)

    def test_custom_assertions(self):
        """Test if custom assertion methods work"""
        # Test assertTensorShape
        test_tensor = torch.randn(2, 3, 4, 4)
        self.assertTensorShape(test_tensor, (2, 3, 4, 4))

        # Test assertTensorInRange
        test_tensor = torch.rand(2, 3, 4, 4)  # Values between 0 and 1
        self.assertTensorInRange(test_tensor, 0, 1)

    def test_random_seed(self):
        """Test if random seeds produce consistent results"""
        torch.manual_seed(42)
        tensor1 = torch.randn(2, 3)

        torch.manual_seed(42)
        tensor2 = torch.randn(2, 3)

        self.assertTrue(torch.equal(tensor1, tensor2))


if __name__ == '__main__':
    unittest.main()