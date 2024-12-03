import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_metrics(psnr_values, ssim_values):
    """Create visualizations of the metrics"""
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Set figure style
    plt.style.use('default')
    
    # Plot PSNR distribution
    plt.figure(figsize=(10, 6))
    plt.hist(psnr_values, bins=15, color='blue', alpha=0.7)
    plt.axvline(np.mean(psnr_values), color='red', linestyle='dashed', linewidth=2)
    plt.text(np.mean(psnr_values)*1.02, plt.ylim()[1]*0.9, 
             f'Mean: {np.mean(psnr_values):.2f} dB')
    plt.title('PSNR Distribution', fontsize=12, pad=15)
    plt.xlabel('PSNR (dB)', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / 'psnr_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot SSIM distribution
    plt.figure(figsize=(10, 6))
    plt.hist(ssim_values, bins=15, color='green', alpha=0.7)
    plt.axvline(np.mean(ssim_values), color='red', linestyle='dashed', linewidth=2)
    plt.text(np.mean(ssim_values)*1.02, plt.ylim()[1]*0.9, 
             f'Mean: {np.mean(ssim_values):.4f}')
    plt.title('SSIM Distribution', fontsize=12, pad=15)
    plt.xlabel('SSIM', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / 'ssim_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(psnr_values, ssim_values, alpha=0.6, c='blue')
    plt.title('PSNR vs SSIM', fontsize=12, pad=15)
    plt.xlabel('PSNR (dB)', fontsize=10)
    plt.ylabel('SSIM', fontsize=10)
    
    # Add trend line
    z = np.polyfit(psnr_values, ssim_values, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(psnr_values), max(psnr_values), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8, label='Trend Line')
    
    # Add correlation coefficient
    correlation = np.corrcoef(psnr_values, ssim_values)[0,1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
             transform=plt.gca().transAxes)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(results_dir / 'psnr_vs_ssim.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"PSNR (dB):")
    print(f"  Mean: {np.mean(psnr_values):.2f}")
    print(f"  Std Dev: {np.std(psnr_values):.2f}")
    print(f"  Min: {np.min(psnr_values):.2f}")
    print(f"  Max: {np.max(psnr_values):.2f}")
    print(f"\nSSIM:")
    print(f"  Mean: {np.mean(ssim_values):.4f}")
    print(f"  Std Dev: {np.std(ssim_values):.4f}")
    print(f"  Min: {np.min(ssim_values):.4f}")
    print(f"  Max: {np.max(ssim_values):.4f}")
    print(f"\nCorrelation between PSNR and SSIM: {correlation:.4f}")

def main():
    # Test results
    psnr_values = [
        30.00, 26.02, 27.87, 28.67, 29.55, 24.62, 34.56, 31.78, 29.41, 27.85,
        36.98, 28.40, 29.68, 28.04, 28.12, 28.28, 26.01, 27.57, 32.62, 28.26,
        29.62, 28.40, 28.89, 26.50, 31.47
    ]
    
    ssim_values = [
        0.8224, 0.7886, 0.7605, 0.8129, 0.8632, 0.7431, 0.9198, 0.8643, 0.6769,
        0.8124, 0.8195, 0.8126, 0.8813, 0.8564, 0.7800, 0.8115, 0.8309, 0.7698,
        0.9453, 0.8126, 0.7443, 0.8680, 0.8845, 0.8603, 0.8584
    ]
    
    visualize_metrics(psnr_values, ssim_values)
    print("\nVisualization completed! Check the results directory for plots.")

if __name__ == "__main__":
    main()
