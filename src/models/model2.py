import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSuperResolutionModel(nn.Module):
    """
    Model-2: Simple CNN architecture for super-resolution
    As per paper: Direct convolutions without down/up-sampling
    Structure: Input -> 64 -> 128 -> 256 -> 128 -> 64 -> 3 filters
    """
    def __init__(self, input_channels=3, l1_factor=1e-10):
        super(SimpleSuperResolutionModel, self).__init__()
        
        # L1 regularization factor
        self.l1_factor = l1_factor
        
        # Sequential model with direct convolutions
        self.model = nn.Sequential(
            # First conv layer: 3 -> 64 channels
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Second conv layer: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Third conv layer: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth conv layer: 256 -> 128 channels
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth conv layer: 128 -> 64 channels
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Output conv layer: 64 -> 3 channels
            nn.Conv2d(64, input_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)
    
    def l1_loss(self):
        """Calculate L1 regularization loss"""
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_factor * l1_loss

def count_parameters(model):
    """Utility function to count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test the model architecture
if __name__ == "__main__":
    # Create model instance
    model = SimpleSuperResolutionModel()
    
    # Print model architecture
    print(model)
    
    # Print number of parameters
    print(f"\nTrainable parameters: {count_parameters(model):,}")
    
    # Test with random input
    test_input = torch.randn(1, 3, 256, 256)
    output = model(test_input)
    print(f"\nOutput shape: {output.shape}")
