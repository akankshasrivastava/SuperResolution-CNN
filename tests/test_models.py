import unittest
import torch
from pathlib import Path
import sys

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from models.model1 import SuperResolutionModel
from models.model2 import SimpleSuperResolutionModel, count_parameters
from tests.test_base import BaseTestCase


class TestSuperResolutionModel(BaseTestCase):
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.model = SuperResolutionModel()
        self.model.eval()

    def test_model_output_shape(self):
        """Test if model preserves input dimensions"""
        with torch.no_grad():
            output = self.model(self.dummy_input)
        self.assertTensorShape(output, (self.batch_size, self.channels, self.height, self.width))

    def test_model_output_range(self):
        """Test if model output is in valid range (after ReLU, should be >= 0)"""
        with torch.no_grad():
            output = self.model(self.dummy_input)
        self.assertTensorInRange(output, 0, float('inf'))

    def test_l1_loss(self):
        """Test if L1 regularization loss is calculated correctly"""
        l1_loss = self.model.l1_loss()
        self.assertIsInstance(l1_loss, torch.Tensor)
        self.assertTrue(l1_loss >= 0)


class TestSimpleSuperResolutionModel(BaseTestCase):
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.model = SimpleSuperResolutionModel()
        self.model.eval()

    def test_model_output_shape(self):
        """Test if model preserves input dimensions"""
        with torch.no_grad():
            output = self.model(self.dummy_input)
        self.assertTensorShape(output, (self.batch_size, self.channels, self.height, self.width))

    def test_model_output_range(self):
        """Test if model output is in valid range"""
        with torch.no_grad():
            output = self.model(self.dummy_input)
        self.assertTensorInRange(output, 0, float('inf'))

    def test_parameter_count(self):
        """Test parameter counting utility"""
        num_params = count_parameters(self.model)
        self.assertGreater(num_params, 0)
        self.assertEqual(num_params, sum(p.numel() for p in self.model.parameters() if p.requires_grad))


if __name__ == '__main__':
    unittest.main()