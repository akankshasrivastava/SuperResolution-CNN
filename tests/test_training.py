import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from models.model1 import SuperResolutionModel
from tests.test_base import BaseTestCase


class TestTraining(BaseTestCase):
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.model = SuperResolutionModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.L1Loss()

    def test_forward_backward_pass(self):
        """Test if model can perform forward and backward passes"""
        # Forward pass
        output = self.model(self.dummy_input)
        loss = self.criterion(output, self.dummy_target)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Check if gradients were computed
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_training_step(self):
        """Test if model weights update during training"""
        # Get initial weights
        initial_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial_weights[name] = param.clone()

        # Perform training step
        self.model.train()
        output = self.model(self.dummy_input)
        loss = self.criterion(output, self.dummy_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Check if weights changed
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertFalse(torch.equal(param, initial_weights[name]))

    def test_gradient_flow(self):
        """Test if gradients flow through all layers"""
        self.model.train()
        output = self.model(self.dummy_input)
        loss = self.criterion(output, self.dummy_target)
        loss.backward()

        # Check gradients in different parts of the model
        conv1_grad = self.model.conv1.weight.grad
        final_conv_grad = self.model.final_conv.weight.grad

        self.assertIsNotNone(conv1_grad)
        self.assertIsNotNone(final_conv_grad)
        self.assertGreater(conv1_grad.abs().sum().item(), 0)
        self.assertGreater(final_conv_grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()