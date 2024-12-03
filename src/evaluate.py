import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from models.model1 import SuperResolutionModel
from data.dataset import SuperResolutionDataset
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, 
               channel_axis=2,  # Specify channel axis
               win_size=3,      # Use smaller window size
               data_range=1.0)  # Specify data range for [0,1] float images

def evaluate_model(test_dataset, model, device):
    """Evaluate model on test dataset"""
    model.eval()
    psnr_values = []
    ssim_values = []
    
    print("\nEvaluating model on test dataset...")
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            # Load and process images
            lr_img, hr_img = test_dataset[idx]
            
            # Process LR image
            lr_img = lr_img.unsqueeze(0).to(device)  # Add batch dimension
            
            # Generate SR image
            sr_img = model(lr_img)
            sr_img = sr_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            sr_img = np.clip(sr_img, 0, 1)
            
            # Convert HR image to numpy
            hr_img = hr_img.cpu().numpy().transpose(1, 2, 0)
            
            # Get LR image for visualization
            lr_vis = lr_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            # Calculate metrics
            curr_psnr = calculate_psnr(hr_img, sr_img)
            curr_ssim = calculate_ssim(hr_img, sr_img)
            
            psnr_values.append(curr_psnr)
            ssim_values.append(curr_ssim)
            
            print(f"Image {idx + 1}: PSNR = {curr_psnr:.2f} dB, SSIM = {curr_ssim:.4f}")
            
            # Save comparison plot for first 5 images
            if idx < 5:
                plot_comparison(hr_img, sr_img, lr_vis, f'result_{idx}', curr_psnr, curr_ssim)
    
    return np.mean(psnr_values), np.mean(ssim_values)

def plot_comparison(hr_img, sr_img, lr_img, name, psnr_val, ssim_val):
    """Plot and save comparison of HR, LR, and SR images"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(hr_img)
    plt.title('High Resolution')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(lr_img)
    plt.title('Low Resolution')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(sr_img)
    plt.title(f'Super Resolution\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}')
    plt.axis('off')
    
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    plt.savefig(results_dir / f'{name}.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load model
        model = SuperResolutionModel().to(device)
        checkpoint = torch.load('checkpoints/best_model.pth', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create test dataset
        test_dataset = SuperResolutionDataset(
            hr_dir='data/test/hr',
            lr_dir='data/test/lr'
        )
        
        print(f"Found {len(test_dataset)} test images")
        
        # Evaluate model
        avg_psnr, avg_ssim = evaluate_model(test_dataset, model, device)
        
        print(f"\nTest Results:")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        
        # Save metrics to file
        with open('results/metrics.txt', 'w') as f:
            f.write(f"Test Results:\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
